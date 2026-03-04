import os
import argparse
import json
import numpy as np
import pandas as pd


def safe_div(a, b) -> float:
    a = float(a)
    b = float(b)
    return a / b if b != 0 else 0.0


def compute_metrics(cm: np.ndarray) -> dict:
    """
    Expects confusion matrix of shape (K+1, K+1) where last index is unknown.
    """
    K_plus_1 = int(cm.shape[0])
    K = K_plus_1 - 1
    UNK = K

    total = float(cm.sum())
    diag_total = float(np.trace(cm))

    # Known rows: 0..K-1
    known_total = float(cm[:K, :].sum())
    known_correct = float(np.trace(cm[:K, :K]))
    known_acc_th = safe_div(known_correct, known_total)

    # Unknown row
    unk_total = float(cm[UNK, :].sum())
    unk_correct = float(cm[UNK, UNK])
    unk_acc_th = safe_div(unk_correct, unk_total)

    # Open-set accuracy (overall diagonal / total)
    os_acc = safe_div(diag_total, total)

    # Unknown F1 (treat unknown as positive class)
    tp = float(cm[UNK, UNK])
    fp = float(cm[:K, UNK].sum())   # known predicted as unknown
    fn = float(cm[UNK, :K].sum())   # unknown predicted as known

    prec_u = safe_div(tp, tp + fp)
    rec_u = safe_div(tp, tp + fn)
    f1_u = safe_div(2 * prec_u * rec_u, prec_u + rec_u) if (prec_u + rec_u) else 0.0

    # Macro-F1 across all (K+1) classes
    f1s = []
    for c in range(K_plus_1):
        tp_c = float(cm[c, c])
        fp_c = float(cm[:, c].sum() - tp_c)
        fn_c = float(cm[c, :].sum() - tp_c)
        prec_c = safe_div(tp_c, tp_c + fp_c)
        rec_c = safe_div(tp_c, tp_c + fn_c)
        f1_c = safe_div(2 * prec_c * rec_c, prec_c + rec_c) if (prec_c + rec_c) else 0.0
        f1s.append(f1_c)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "Known Class Accuracy (Th)": known_acc_th,
        "Unknown Class Accuracy (Th)": unk_acc_th,
        "Open-Set Accuracy": os_acc,
        "F1 (Unknown)": f1_u,
        "Macro F1": macro_f1,
    }


def format_percent_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = (out[col] * 100.0).map(lambda x: f"{x:.2f}%")
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare OSR metrics from *_logits_best_confusion.npy matrices.")
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing confusion matrices (e.g., output/models/resnet18_2).",
    )
    ap.add_argument(
        "--methods",
        nargs="*",
        default=["openmax", "mls", "mahalanobis"],
        help="Which methods to include (default: openmax mls mahalanobis).",
    )
    ap.add_argument(
        "--suffix",
        default="logits_best_confusion.npy",
        help="Filename suffix (default: logits_best_confusion.npy).",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path. Default: <model-dir>/osr_comparison_table.csv",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON path with raw floats. Default: <model-dir>/osr_comparison_table.json",
    )
    args = ap.parse_args()

    model_dir = args.model_dir
    methods = args.methods
    suffix = args.suffix

    rows = []
    missing = []

    for m in methods:
        path = os.path.join(model_dir, f"{m}_{suffix}")
        if not os.path.exists(path):
            missing.append(path)
            continue
        cm = np.load(path)
        metrics = compute_metrics(cm)
        metrics["Model"] = m
        rows.append(metrics)

    if not rows:
        raise SystemExit(f"No confusion matrices found. Missing:\n" + "\n".join(missing))

    df = pd.DataFrame(rows).set_index("Model")
    df_pct = format_percent_df(df)

    print("\n===== OSR Comparison Table =====\n")
    print(df_pct.to_string())

    out_csv = args.out_csv or os.path.join(model_dir, "osr_comparison_table.csv")
    df_pct.to_csv(out_csv)

    out_json = args.out_json or os.path.join(model_dir, "osr_comparison_table.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(df.reset_index().to_dict(orient="records"), f, indent=2)

    print("\nSaved CSV to:", out_csv)
    print("Saved JSON to:", out_json)

    if missing:
        print("\nNote: Missing matrices (skipped):")
        for p in missing:
            print(" -", p)


if __name__ == "__main__":
    main()