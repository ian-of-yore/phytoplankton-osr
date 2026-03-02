import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def tune_threshold_mls(mls_val, val_y_u, K, thresh_grid, target_known_recall=0.95):
    val_is_unknown = (val_y_u == K)
    mask_known = ~val_is_unknown

    best = None  # (unk_f1, t, known_recall, unk_recall)
    for t in thresh_grid:
        pred_unk = (mls_val < t)  # low MLS => unknown
        known_recall = 1.0 - pred_unk[mask_known].mean()
        unk_recall = pred_unk[val_is_unknown].mean() if np.any(val_is_unknown) else 0.0
        unk_f1 = f1_score(val_is_unknown.astype(int), pred_unk.astype(int), zero_division=0)

        if known_recall >= target_known_recall:
            if (best is None) or (unk_f1 > best[0]):
                best = (float(unk_f1), float(t), float(known_recall), float(unk_recall))

    if best is None:
        # fallback: 1% quantile of known MLS
        best_t = float(np.quantile(mls_val[mask_known], 0.01)) if np.any(mask_known) else float(np.quantile(mls_val, 0.01))
        return best_t, {"note": "fallback", "target_known_recall": target_known_recall}
    return best[1], {"unk_f1": best[0], "known_recall": best[2], "unk_recall": best[3], "target_known_recall": target_known_recall}

def evaluate_mls(test_logits, test_y_u, K, best_t, known_classes, unknown_name="__unknown__"):
    mls_test = np.max(test_logits, axis=1)
    pred_unk_test = (mls_test < best_t)

    pred_known_base = np.argmax(test_logits, axis=1)
    pred_u = pred_known_base.copy()
    pred_u[pred_unk_test] = K

    target_names = list(known_classes) + [unknown_name]
    rep = classification_report(test_y_u, pred_u, target_names=target_names, digits=3, zero_division=0)
    cm = confusion_matrix(test_y_u, pred_u, labels=list(range(K+1)))

    mask_known_test = (test_y_u != K)
    known_acc = accuracy_score(test_y_u[mask_known_test], pred_u[mask_known_test]) if np.any(mask_known_test) else float("nan")
    unk_recall = (pred_u[test_y_u == K] == K).mean() if np.any(test_y_u == K) else float("nan")

    return pred_u, cm, rep, {"known_acc_th": known_acc, "unknown_recall": unk_recall}
