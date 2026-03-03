# scripts/run_osr.py

import argparse
import os
import sys
import yaml
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phytosr.datasets import make_loader, extract_logits
from phytosr.eval import summarize_osr_from_cm, known_only_cm
from phytosr.plots import plot_cm_counts, plot_cm_row_normalized, plot_known_only_paper


def load_sykepic_model(model_dir: str, sykepic_repo: str = ""):
    """
    Option A friendly:
    - If sykepic_repo is provided: import sykepic from that local path.
    - Else: import sykepic from the installed package (pip).
    """
    if sykepic_repo:
        if sykepic_repo not in sys.path:
            sys.path.insert(0, sykepic_repo)

    from sykepic.compute.probability import prepare_model

    net, classes, img_shape, eval_transform, device = prepare_model(model_dir)
    net = net.to(device).eval()
    return net, list(classes), eval_transform, device


def resolve_model_dir(cfg):
    # If user sets explicit model_dir, use it
    if "model_dir" in cfg["paths"] and cfg["paths"]["model_dir"]:
        return cfg["paths"]["model_dir"]

    # else try pointer from model_out_dir
    model_out_dir = cfg["paths"]["model_out_dir"]
    pointer = os.path.join(model_out_dir, "metadata", "latest_model_dir.txt")
    if os.path.exists(pointer):
        return open(pointer, "r", encoding="utf-8").read().strip()

    raise RuntimeError("MODEL_DIR not found. Set paths.model_dir or ensure latest_model_dir.txt exists.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Experiment config (e.g., configs/exp.yaml)")
    ap.add_argument("--paths", required=True, help="Machine-specific paths (e.g., configs/paths.yaml)")
    ap.add_argument("--method", required=True, choices=["openmax", "mls", "mahalanobis"])
    args = ap.parse_args()

    # Load experiment config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load machine-specific paths and override cfg["paths"]
    with open(args.paths, "r", encoding="utf-8") as f:
        paths_cfg = yaml.safe_load(f)
    cfg["paths"] = paths_cfg["paths"]

    method = args.method

    out_base = cfg["paths"]["out_base"]
    train_root = os.path.join(out_base, "train_root")
    osr_val_root = os.path.join(out_base, "osr_val_root")
    osr_test_root = os.path.join(out_base, "osr_test_root")
    unknown_name = cfg.get("data_prep", {}).get("unknown_name", "__unknown__")

    model_dir = resolve_model_dir(cfg)
    sykepic_repo = cfg["paths"].get("sykepic_repo", "")

    net, known_classes, eval_transform, device = load_sykepic_model(model_dir, sykepic_repo)
    K = len(known_classes)

    bs = int(cfg.get("runtime", {}).get("batch_size", 128))
    nw = int(cfg.get("runtime", {}).get("num_workers", 2))

    # loaders
    train_ds, train_loader = make_loader(
        train_root, known_classes, eval_transform,
        batch_size=bs, num_workers=nw,
        include_unknown=False, unknown_name=unknown_name
    )
    val_ds, val_loader = make_loader(
        osr_val_root, known_classes, eval_transform,
        batch_size=bs, num_workers=nw,
        include_unknown=True, unknown_name=unknown_name
    )
    test_ds, test_loader = make_loader(
        osr_test_root, known_classes, eval_transform,
        batch_size=bs, num_workers=nw,
        include_unknown=True, unknown_name=unknown_name
    )

    print("MODEL_DIR:", model_dir)
    print("Train classes present:", train_ds.classes_present)
    print("Val classes present:", val_ds.classes_present)
    print("Test classes present:", test_ds.classes_present)

    # logits
    train_logits, train_y, _ = extract_logits(net, train_loader, device)
    val_logits, val_y_u, _ = extract_logits(net, val_loader, device)
    test_logits, test_y_u, _ = extract_logits(net, test_loader, device)

    labels = list(known_classes) + [unknown_name]

    plots_dir = os.path.join(model_dir, "plots")
    reports_dir = os.path.join(model_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if method == "openmax":
        from phytosr.methods.openmax_logits import (
            fit_mavs_weibulls,
            openmax_probs_logits,
            tune_threshold_punknown_constrained,
            tune_threshold_punknown_balanced,
            evaluate_openmax,
            save_openmax_artifacts,
        )

        # -------------------------
        # params
        # -------------------------
        om_cfg = cfg.get("openmax", {}) or {}

        tailsize = int(om_cfg.get("tailsize", 20))
        alpha = int(om_cfg.get("alpha", 10))
        euclid = bool(om_cfg.get("use_euclidean", True))

        mode = str(om_cfg.get("mode", "constrained")).lower()
        target_known_recall = float(om_cfg.get("target_known_recall", 0.95))

        # optional: freeze threshold to match notebook exactly (no drift)
        fixed_t = om_cfg.get("fixed_threshold", None)

        # -------------------------
        # fit + probs
        # -------------------------
        mavs, weibulls = fit_mavs_weibulls(train_logits, train_y, K, tailsize=tailsize, euclidean=euclid)
        mav_path, wb_path = save_openmax_artifacts(model_dir, mavs, weibulls, tailsize, alpha, euclid)
        print("Saved OpenMax artifacts:", mav_path, wb_path)

        val_p = openmax_probs_logits(val_logits, mavs, weibulls, alpha=alpha, euclidean=euclid)
        test_p = openmax_probs_logits(test_logits, mavs, weibulls, alpha=alpha, euclidean=euclid)
        u_val = val_p[:, -1]
        u_test = test_p[:, -1]

        # -------------------------
        # choose threshold
        # -------------------------
        if fixed_t is not None:
            best_t = float(fixed_t)
            tune_info = {"note": "fixed_threshold", "best_t": best_t}
        else:
            if mode.startswith("bal"):
                best_t, tune_info = tune_threshold_punknown_balanced(
                    val_logits=val_logits,
                    u_val=u_val,
                    val_y_u=val_y_u,
                    K=K,
                    grid_n=int(om_cfg.get("balanced_grid_n", 401)),
                )
            else:
                best_t, tune_info = tune_threshold_punknown_constrained(
                    u_val=u_val,
                    val_y_u=val_y_u,
                    K=K,
                    target_known_recall=float(target_known_recall),
                    thresh_grid_n=int(om_cfg.get("thresh_grid_n", 201)),
                    fallback_t=float(om_cfg.get("fallback_t", 0.5)),
                )

        print("OpenMax mode:", mode, "| fixed_threshold:", fixed_t)
        print("Threshold:", best_t, "| tune:", tune_info)

        # -------------------------
        # eval
        # -------------------------
        pred_u, cm, rep, metrics = evaluate_openmax(
            test_logits, test_y_u, K, best_t, u_test, known_classes, unknown_name=unknown_name
        )

        cm_path = os.path.join(model_dir, "openmax_logits_best_confusion.npy")
        np.save(cm_path, cm)
        np.save(os.path.join(model_dir, "openmax_known_only_confusion.npy"), known_only_cm(cm, K))
        print("Saved:", cm_path)

        summ = summarize_osr_from_cm(cm, K)
        report_path = os.path.join(reports_dir, f"openmax_report_t{tailsize}_a{alpha}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL_DIR: {model_dir}\n")
            f.write(f"OpenMax logits: tailsize={tailsize}, alpha={alpha}, euclidean={euclid}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Fixed threshold: {fixed_t}\n")
            f.write(f"Threshold: {best_t}\nTune: {tune_info}\n\n")
            f.write(rep + "\n")
            f.write("\nMetrics:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\nCM summary:\n")
            for k, v in summ.items():
                f.write(f"{k}: {v}\n")
        print("Saved report:", report_path)

        plot_cm_counts(
            cm, labels, os.path.join(plots_dir, "openmax_confusion_counts.png"),
            "OpenMax (logits) Confusion Matrix — Counts (OSR test)"
        )
        plot_cm_row_normalized(
            cm, labels, os.path.join(plots_dir, "openmax_confusion_row_normalized.png"),
            "OpenMax (logits) Confusion Matrix — Row-normalized (Recall) (OSR test)"
        )
        plot_known_only_paper(
            known_only_cm(cm, K), known_classes,
            os.path.join(plots_dir, "openmax_known_only_confusion_YlGnBu.png"),
            title="OpenMax Known-only Confusion Matrix (Counts)",
            vmax=None
        )

    elif method == "mls":
        from phytosr.methods.mls import tune_threshold_mls, evaluate_mls

        target_known_recall = float(cfg["mls"].get("target_known_recall", 0.95))
        ngrid = int(cfg["mls"].get("thresh_grid_n", 501))

        mls_val = np.max(val_logits, axis=1)
        # threshold grid based on distribution (unique to avoid repeats)
        thresh_grid = np.unique(np.quantile(mls_val, np.linspace(0.0, 1.0, ngrid)))

        best_t, tune_info = tune_threshold_mls(
            mls_val, val_y_u, K, thresh_grid, target_known_recall=target_known_recall
        )
        print("Threshold:", best_t, "| tune:", tune_info)

        pred_u, cm, rep, metrics = evaluate_mls(
            test_logits, test_y_u, K, best_t, known_classes, unknown_name=unknown_name
        )

        cm_path = os.path.join(model_dir, "mls_logits_best_confusion.npy")
        np.save(cm_path, cm)
        np.save(os.path.join(model_dir, "mls_known_only_confusion.npy"), known_only_cm(cm, K))
        print("Saved:", cm_path)

        summ = summarize_osr_from_cm(cm, K)
        report_path = os.path.join(reports_dir, "mls_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL_DIR: {model_dir}\n")
            f.write(f"MLS threshold: {best_t}\nTune: {tune_info}\n\n")
            f.write(rep + "\n")
            f.write("\nCM summary:\n")
            for k, v in summ.items():
                f.write(f"{k}: {v}\n")
        print("Saved report:", report_path)

        plot_cm_counts(
            cm, labels, os.path.join(plots_dir, "mls_confusion_counts.png"),
            "MLS Confusion Matrix — Counts (OSR test)"
        )
        plot_cm_row_normalized(
            cm, labels, os.path.join(plots_dir, "mls_confusion_row_normalized.png"),
            "MLS Confusion Matrix — Row-normalized (Recall) (OSR test)"
        )
        plot_known_only_paper(
            known_only_cm(cm, K), known_classes,
            os.path.join(plots_dir, "mls_known_only_confusion_YlGnBu.png"),
            title="MLS Known-only Confusion Matrix (Counts)",
            vmax=None
        )

    elif method == "mahalanobis":
        from phytosr.methods.mahalanobis import (
            extract_logits_and_embeddings,
            fit_mahalanobis,
            score_knownness,
            tune_threshold_balanced,
            compute_val_auroc,
            apply_threshold,
        )

        shrink = float(cfg.get("mahalanobis", {}).get("shrink", 0.05))
        n_sweep = int(cfg.get("mahalanobis", {}).get("n_sweep", 300))

        # Extract embeddings using the SAME hook as your notebook
        train_logits2, train_emb, train_y2, _ = extract_logits_and_embeddings(net, train_loader, device)
        val_logits2, val_emb, val_y2, _ = extract_logits_and_embeddings(net, val_loader, device)
        test_logits2, test_emb, test_y2, _ = extract_logits_and_embeddings(net, test_loader, device)

        # Fit stats on train
        mus, prec = fit_mahalanobis(train_emb, train_y2, K, shrink=shrink)

        # Score val/test
        val_score, val_pred_class = score_knownness(val_emb, mus, prec)
        test_score, test_pred_class = score_knownness(test_emb, mus, prec)

        # AUROC (diagnostic, like notebook)
        try:
            auroc = compute_val_auroc(val_score, val_y2, K)
        except Exception as e:
            auroc = None
            print("AUROC could not be computed:", e)

        # Threshold tuning on OSR-val (balanced metric)
        best = tune_threshold_balanced(val_score, val_pred_class, val_y2, K, n_sweep=n_sweep)
        t_star = best["t"]
        print("Val best threshold:", best, "| AUROC:", auroc)

        # Apply OSR rule on OSR-test
        test_pred = apply_threshold(test_score, test_pred_class, t_star, K)

        # Build confusion + report
        from sklearn.metrics import classification_report, confusion_matrix

        target_names = list(known_classes) + [unknown_name]
        rep = classification_report(test_y2, test_pred, target_names=target_names, digits=3, zero_division=0)
        cm = confusion_matrix(test_y2, test_pred, labels=list(range(K + 1)))

        cm_path = os.path.join(model_dir, "mahalanobis_logits_best_confusion.npy")
        np.save(cm_path, cm)
        np.save(os.path.join(model_dir, "mahalanobis_known_only_confusion.npy"), known_only_cm(cm, K))
        print("Saved:", cm_path)

        summ = summarize_osr_from_cm(cm, K)
        report_path = os.path.join(reports_dir, "mahalanobis_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL_DIR: {model_dir}\n")
            f.write(f"Mahalanobis: shrink={shrink}\n")
            f.write(f"Val threshold selection (balanced): {best}\n")
            f.write(f"Val AUROC (unknown vs known) using (-score): {auroc}\n\n")
            f.write(rep + "\n")
            f.write("\nCM summary:\n")
            for k, v in summ.items():
                f.write(f"{k}: {v}\n")
        print("Saved report:", report_path)

        # Plots
        plot_cm_counts(
            cm, labels, os.path.join(plots_dir, "mahalanobis_confusion_counts.png"),
            "Mahalanobis Confusion Matrix — Counts (OSR test)"
        )
        plot_cm_row_normalized(
            cm, labels, os.path.join(plots_dir, "mahalanobis_confusion_row_normalized.png"),
            "Mahalanobis Confusion Matrix — Row-normalized (Recall) (OSR test)"
        )
        plot_known_only_paper(
            known_only_cm(cm, K), known_classes,
            os.path.join(plots_dir, "mahalanobis_known_only_confusion_YlGnBu.png"),
            title="Mahalanobis Known-only Confusion Matrix (Counts)",
            vmax=None
        )

    print("\n✅ Done:", method)


if __name__ == "__main__":
    main()