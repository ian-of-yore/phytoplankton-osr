# src/phytosr/methods/openmax_logits.py

import os
import sys
import pickle
from typing import Any, Dict, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from scipy.stats import weibull_min


# -------------------------
# Dataset (matches your notebooks)
# -------------------------
class CV2FolderOSR(Dataset):
    """
    Folder structure:
      root/
        ClassA/*.png
        ClassB/*.png
        __unknown__/*.png    (optional)

    Labels:
      known classes => 0..K-1 based on SykePic class order
      __unknown__   => K
    """

    def __init__(
        self,
        root: str,
        known_class_order: List[str],
        transform,
        include_unknown: bool,
        unknown_name: str,
    ):
        self.root = root
        self.transform = transform
        self.known_order = list(known_class_order)
        self.class_to_idx = {c: i for i, c in enumerate(self.known_order)}
        self.include_unknown = include_unknown
        self.unknown_name = unknown_name
        self.K = len(self.known_order)

        self.samples: List[Tuple[str, int]] = []
        self.classes_present: List[str] = []

        for cname in sorted(os.listdir(root)):
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir):
                continue

            if cname == self.unknown_name:
                if not include_unknown:
                    continue
                y = self.K
            else:
                if cname not in self.class_to_idx:
                    continue
                y = self.class_to_idx[cname]

            self.classes_present.append(cname)

            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(cdir, fn), y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img) if self.transform else img
        return x, y


def _make_loader(
    root: str,
    known_classes: List[str],
    eval_transform,
    include_unknown: bool,
    unknown_name: str,
    batch_size: int,
    num_workers: int,
):
    ds = CV2FolderOSR(
        root=root,
        known_class_order=known_classes,
        transform=eval_transform,
        include_unknown=include_unknown,
        unknown_name=unknown_name,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, dl


@torch.no_grad()
def _extract_logits(net, loader, device: str):
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = net(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(np.asarray(y))
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


# -------------------------
# OpenMax core (logit-space)
# -------------------------
def fit_mavs_weibulls(
    train_logits: np.ndarray,
    train_y: np.ndarray,
    K: int,
    tailsize: int,
    euclidean: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mavs: (K, K)      mean logit vector per class
      weibulls: (K, 2)  (shape, scale) per class
    """
    mavs = np.zeros((K, K), dtype=np.float64)
    for c in range(K):
        Xc = train_logits[train_y == c]
        if Xc.shape[0] == 0:
            raise RuntimeError(f"No train samples for class index {c}")
        mavs[c] = Xc.mean(axis=0)

    weibulls = []
    for c in range(K):
        Xc = train_logits[train_y == c]

        if euclidean:
            d = np.linalg.norm(Xc - mavs[c], axis=1)
        else:
            # cosine distance in logit space
            Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
            mn = mavs[c] / (np.linalg.norm(mavs[c]) + 1e-12)
            d = 1.0 - (Xn @ mn)

        d = np.sort(d)[::-1]  # largest first
        tail = d[: min(tailsize, len(d))]
        shape, loc, scale = weibull_min.fit(tail, floc=0.0)
        weibulls.append((float(shape), float(scale)))

    weibulls = np.array(weibulls, dtype=np.float64)
    return mavs, weibulls


def openmax_probs_logits(
    logits: np.ndarray,
    mavs: np.ndarray,
    weibulls: np.ndarray,
    alpha: int = 10,
    euclidean: bool = True,
) -> np.ndarray:
    """
    logits: (N, K)
    returns: (N, K+1) probabilities over K known + 1 unknown
    """
    N, K = logits.shape
    out = np.zeros((N, K + 1), dtype=np.float32)

    for i in range(N):
        x = logits[i].astype(np.float64)
        top_idx = np.argsort(x)[::-1][:alpha]

        revised = x.copy()
        unknown_mass = 0.0

        if not euclidean:
            xn = x / (np.linalg.norm(x) + 1e-12)

        for c in top_idx:
            shape, scale = weibulls[c]

            if euclidean:
                d = float(np.linalg.norm(x - mavs[c]))
            else:
                mn = mavs[c] / (np.linalg.norm(mavs[c]) + 1e-12)
                d = float(1.0 - np.dot(xn, mn))

            w = weibull_min.sf(d, shape, loc=0.0, scale=scale)
            delta = revised[c] * (1.0 - w)
            revised[c] = revised[c] * w
            unknown_mass += delta

        ext = np.concatenate([revised, np.array([unknown_mass], dtype=np.float64)])
        ext = ext - np.max(ext)
        p = np.exp(ext) / np.sum(np.exp(ext))
        out[i] = p.astype(np.float32)

    return out


# -------------------------
# Threshold tuning (two modes)
# -------------------------
def tune_threshold_punknown_constrained(
    u_val: np.ndarray,
    val_y_u: np.ndarray,
    K: int,
    target_known_recall: float,
    thresh_grid_n: int,
    fallback_t: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Constrained tuning:
      - grid t in [0,1]
      - choose t maximizing unknown F1, subject to known_recall >= target
      - if none, fallback to fallback_t
    """
    val_is_unknown = (val_y_u == K)
    mask_known = ~val_is_unknown

    THRESH_GRID = np.linspace(0.0, 1.0, int(thresh_grid_n))
    best = None  # (unk_f1, t, known_recall, unk_recall)

    for t in THRESH_GRID:
        pred_unk = (u_val >= t)
        known_recall = 1.0 - pred_unk[mask_known].mean() if mask_known.any() else 0.0
        unk_recall = pred_unk[val_is_unknown].mean() if val_is_unknown.any() else 0.0
        unk_f1 = f1_score(val_is_unknown.astype(int), pred_unk.astype(int), zero_division=0)

        if known_recall >= target_known_recall:
            if (best is None) or (unk_f1 > best[0]):
                best = (float(unk_f1), float(t), float(known_recall), float(unk_recall))

    if best is None:
        return float(fallback_t), {"note": "fallback", "target_known_recall": float(target_known_recall)}

    unk_f1_val, best_t, known_rec_val, unk_rec_val = best
    return float(best_t), {
        "note": "constraint",
        "target_known_recall": float(target_known_recall),
        "unk_f1_val": float(unk_f1_val),
        "known_recall_val": float(known_rec_val),
        "unk_recall_val": float(unk_rec_val),
    }


def tune_threshold_punknown_balanced(
    val_logits: np.ndarray,
    u_val: np.ndarray,
    val_y_u: np.ndarray,
    K: int,
    grid_n: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    Balanced objective:
      score(t) = 0.5*(known_acc(t) + unk_rec(t))
    """
    THRESH_GRID = np.linspace(0.0, 1.0, int(grid_n))
    best = None  # (score, t, known_acc, unk_rec)

    for t in THRESH_GRID:
        pred_unk = (u_val >= t)
        pred = np.argmax(val_logits, axis=1).copy()
        pred[pred_unk] = K

        known_mask = (val_y_u != K)
        unk_mask = (val_y_u == K)

        known_acc = accuracy_score(val_y_u[known_mask], pred[known_mask]) if known_mask.any() else np.nan
        unk_rec = (pred[unk_mask] == K).mean() if unk_mask.any() else np.nan

        score = 0.5 * (known_acc + unk_rec) if (not np.isnan(known_acc) and not np.isnan(unk_rec)) else -1.0
        if (best is None) or (score > best[0]):
            best = (float(score), float(t), float(known_acc), float(unk_rec))

    score, best_t, known_acc, unk_rec = best
    return float(best_t), {
        "note": "balanced",
        "val_score": float(score),
        "val_known_acc": float(known_acc),
        "val_unk_rec": float(unk_rec),
        "grid_n": int(grid_n),
    }


# -------------------------
# Eval + Saving
# -------------------------
def evaluate_openmax(
    test_logits: np.ndarray,
    test_y_u: np.ndarray,
    K: int,
    best_t: float,
    u_test: np.ndarray,
    known_classes: List[str],
    unknown_name: str = "__unknown__",
):
    """
    Returns (compatible with scripts/run_osr.py):
      pred_u, cm, rep, metrics
    """
    pred_unk_test = (u_test >= best_t)
    pred_known_base = np.argmax(test_logits, axis=1)
    pred_u = pred_known_base.copy()
    pred_u[pred_unk_test] = K

    target_names = list(known_classes) + [unknown_name]
    rep = classification_report(test_y_u, pred_u, target_names=target_names, digits=3, zero_division=0)
    cm = confusion_matrix(test_y_u, pred_u, labels=list(range(K + 1)))

    mask_known_test = (test_y_u != K)
    known_acc_th = (
        accuracy_score(test_y_u[mask_known_test], pred_u[mask_known_test])
        if mask_known_test.any()
        else float("nan")
    )
    unknown_recall = (pred_u[test_y_u == K] == K).mean() if (test_y_u == K).any() else float("nan")

    metrics = {
        "known_acc_th": float(known_acc_th),
        "unknown_recall": float(unknown_recall),
        "best_t": float(best_t),
    }
    return pred_u, cm, rep, metrics


def evaluate_openmax_dict(
    test_logits: np.ndarray,
    test_y_u: np.ndarray,
    K: int,
    best_t: float,
    u_test: np.ndarray,
    known_classes: List[str],
    unknown_name: str = "__unknown__",
) -> Dict[str, Any]:
    """
    Dict-returning evaluator for the internal run_openmax_logits() entrypoint.
    """
    pred_u, cm, rep, metrics = evaluate_openmax(
        test_logits=test_logits,
        test_y_u=test_y_u,
        K=K,
        best_t=best_t,
        u_test=u_test,
        known_classes=known_classes,
        unknown_name=unknown_name,
    )
    return {
        "pred": pred_u,
        "cm": cm,
        "rep": rep,
        "known_acc_th": metrics["known_acc_th"],
        "unknown_recall": metrics["unknown_recall"],
        "best_t": metrics["best_t"],
    }


def save_openmax_artifacts(
    model_dir: str,
    mavs: np.ndarray,
    weibulls: np.ndarray,
    tailsize: int,
    alpha: int,
    euclidean: bool,
):
    out_mavs = os.path.join(model_dir, "openmax_mavs_logits_best.npy")
    out_wb = os.path.join(model_dir, "openmax_weibull_logits_best.pkl")
    np.save(out_mavs, mavs)
    with open(out_wb, "wb") as f:
        pickle.dump(
            {
                "weibulls": np.asarray(weibulls, dtype=np.float64),
                "tailsize": int(tailsize),
                "alpha": int(alpha),
                "euclidean": bool(euclidean),
                "dist": "euclidean" if euclidean else "cosine",
            },
            f,
        )
    return out_mavs, out_wb


# -------------------------
# MAIN ENTRYPOINT (optional)
# -------------------------
def run_openmax_logits(
    model_dir: str,
    train_root: str,
    osr_val_root: str,
    osr_test_root: str,
    sykepic_repo: str = "/content/syke-pic",
    unknown_name: str = "__unknown__",
    batch_size: int = 128,
    num_workers: int = 2,
    # mode:
    #   "constrained" -> constrained tuning on [0,1] grid with fallback
    #   "balanced"    -> balanced tuning
    mode: str = "constrained",
    # OpenMax params:
    tailsize: int = 20,
    alpha: int = 10,
    use_euclidean: bool = True,
    # constrained tuning params:
    target_known_recall: float = 0.95,
    thresh_grid_n: int = 201,
    fallback_t: float = 0.5,
    # balanced tuning params:
    balanced_grid_n: int = 401,
    # saving
    save_confusion_name: str = "openmax_logits_best_confusion.npy",
    save_report_name: str = "openmax_logits_best_report.txt",
    save_best_artifacts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Standalone runner (kept for completeness). scripts/run_osr.py is the preferred interface.
    """
    if sykepic_repo and sykepic_repo not in sys.path:
        sys.path.insert(0, sykepic_repo)

    from sykepic.compute.probability import prepare_model

    net, classes, img_shape, eval_transform, dev = prepare_model(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device).eval()

    known_classes = list(classes)
    K = len(known_classes)

    train_ds, train_loader = _make_loader(
        train_root,
        known_classes,
        eval_transform,
        include_unknown=False,
        unknown_name=unknown_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_ds, val_loader = _make_loader(
        osr_val_root,
        known_classes,
        eval_transform,
        include_unknown=True,
        unknown_name=unknown_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_ds, test_loader = _make_loader(
        osr_test_root,
        known_classes,
        eval_transform,
        include_unknown=True,
        unknown_name=unknown_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if verbose:
        print("Known classes (K):", K)
        print("device:", device)
        print("Train classes present:", getattr(train_ds, "classes_present", None))
        print("Val classes present:", getattr(val_ds, "classes_present", None))
        print("Test classes present:", getattr(test_ds, "classes_present", None))

    train_logits, train_y = _extract_logits(net, train_loader, device)
    val_logits, val_y_u = _extract_logits(net, val_loader, device)
    test_logits, test_y_u = _extract_logits(net, test_loader, device)

    if train_logits.shape[1] != K:
        raise RuntimeError(f"Logit dimension mismatch: train_logits.shape[1]={train_logits.shape[1]} but K={K}")

    mavs, weibulls = fit_mavs_weibulls(
        train_logits=train_logits,
        train_y=train_y,
        K=K,
        tailsize=int(tailsize),
        euclidean=bool(use_euclidean),
    )

    val_p = openmax_probs_logits(val_logits, mavs, weibulls, alpha=int(alpha), euclidean=bool(use_euclidean))
    test_p = openmax_probs_logits(test_logits, mavs, weibulls, alpha=int(alpha), euclidean=bool(use_euclidean))
    u_val = val_p[:, -1]
    u_test = test_p[:, -1]

    val_is_unknown = (val_y_u == K)
    if verbose:
        if val_is_unknown.any():
            print("AUROC (unknown vs known) using P_unknown:", roc_auc_score(val_is_unknown.astype(int), u_val))
        print("P_unknown quantiles (ALL):", np.quantile(u_val, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))

    mode_l = str(mode).lower()
    if mode_l.startswith("bal"):
        best_t, tune_info = tune_threshold_punknown_balanced(
            val_logits=val_logits,
            u_val=u_val,
            val_y_u=val_y_u,
            K=K,
            grid_n=int(balanced_grid_n),
        )
    else:
        best_t, tune_info = tune_threshold_punknown_constrained(
            u_val=u_val,
            val_y_u=val_y_u,
            K=K,
            target_known_recall=float(target_known_recall),
            thresh_grid_n=int(thresh_grid_n),
            fallback_t=float(fallback_t),
        )

    eval_out = evaluate_openmax_dict(
        test_logits=test_logits,
        test_y_u=test_y_u,
        K=K,
        best_t=float(best_t),
        u_test=u_test,
        known_classes=known_classes,
        unknown_name=unknown_name,
    )

    os.makedirs(model_dir, exist_ok=True)
    cm_path = os.path.join(model_dir, save_confusion_name)
    report_path = os.path.join(model_dir, save_report_name)

    np.save(cm_path, eval_out["cm"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("OpenMax (logits)\n")
        f.write(f"MODEL_DIR={model_dir}\n")
        f.write(f"TRAIN_ROOT={train_root}\nOSR_VAL_ROOT={osr_val_root}\nOSR_TEST_ROOT={osr_test_root}\n\n")
        f.write(f"params: tailsize={tailsize}, alpha={alpha}, euclidean={use_euclidean}\n")
        f.write(f"tuning: mode={mode}, tune_info={tune_info}\n")
        f.write(f"best_t={best_t}\n\n")
        f.write(eval_out["rep"] + "\n")
        f.write(f"\nKnown-only accuracy (with threshold): {eval_out['known_acc_th']:.6f}\n")
        f.write(f"Unknown recall: {eval_out['unknown_recall']:.6f}\n")

    artifact_paths = {}
    if save_best_artifacts:
        out_mavs, out_wb = save_openmax_artifacts(
            model_dir, mavs, weibulls, tailsize=int(tailsize), alpha=int(alpha), euclidean=bool(use_euclidean)
        )
        artifact_paths = {"mavs": out_mavs, "weibull": out_wb}

    info = {
        "best_t": float(best_t),
        "tune_info": tune_info,
        "known_acc_th": float(eval_out["known_acc_th"]),
        "unknown_recall": float(eval_out["unknown_recall"]),
        "saved_cm": cm_path,
        "saved_report": report_path,
        "saved_artifacts": artifact_paths,
        "K": int(K),
        "classes": known_classes,
    }

    if verbose:
        print("✅ OpenMax finished")
        print("best_t:", info["best_t"])
        print("tune_info:", info["tune_info"])
        print("known_acc_th:", info["known_acc_th"])
        print("unknown_recall:", info["unknown_recall"])
        print("saved cm:", info["saved_cm"])
        print("saved report:", info["saved_report"])
        if artifact_paths:
            print("saved mavs:", artifact_paths["mavs"])
            print("saved weibull:", artifact_paths["weibull"])

    return info