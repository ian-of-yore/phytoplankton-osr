import numpy as np

def summarize_osr_from_cm(cm: np.ndarray, K: int) -> dict:
    """
    cm shape: (K+1, K+1)
    K: number of known classes
    """
    total = float(cm.sum())
    diag = np.diag(cm).astype(float)

    known_total = float(cm[:K, :].sum())
    known_correct = float(diag[:K].sum())
    known_acc = known_correct / known_total if known_total > 0 else float("nan")

    unk_total = float(cm[K, :].sum())
    unk_correct = float(cm[K, K])
    unk_recall = unk_correct / unk_total if unk_total > 0 else float("nan")

    false_reject = float(cm[:K, K].sum()) / known_total if known_total > 0 else float("nan")
    false_accept = float(cm[K, :K].sum()) / unk_total if unk_total > 0 else float("nan")

    return {
        "total": total,
        "known_acc_th": known_acc,
        "unknown_recall": unk_recall,
        "false_reject_known_to_unknown": false_reject,
        "false_accept_unknown_to_known": false_accept,
    }


def known_only_cm(cm: np.ndarray, K: int) -> np.ndarray:
    return cm[:K, :K].copy()
