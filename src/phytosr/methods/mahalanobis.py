import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

@torch.no_grad()
def extract_logits_and_embeddings(model, loader, device):
    """
    Matches your notebook:
      - hook model.base[8] (AdaptiveAvgPool2d)
      - feat shape: [B, 512, 1, 1] -> flatten -> [B, 512]
    Returns:
      logits: (N, K)
      emb: (N, D)
      y: (N,)
      paths: list[str]
    """
    logits_all, feats_all, y_all, paths_all = [], [], [], []

    captured = []
    def hook_fn(module, inp, out):
        captured.append(out.detach())

    handle = model.base[8].register_forward_hook(hook_fn)

    for x, y, paths in loader:
        captured.clear()
        x = x.to(device, non_blocking=True)

        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        feat = captured[0]            # [B, 512, 1, 1]
        feat = torch.flatten(feat, 1) # [B, 512]

        logits_all.append(logits.detach().cpu().numpy())
        feats_all.append(feat.cpu().numpy())
        y_all.append(np.array(y))
        paths_all.extend(list(paths))

    handle.remove()

    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(feats_all, axis=0),
        np.concatenate(y_all, axis=0),
        paths_all
    )

def fit_mahalanobis(train_emb, train_y, K, shrink=0.05):
    """
    Same as your notebook:
      mus[c] = mean embedding of class c
      shared covariance pooled across all samples
      shrinkage: (1-s)*cov + s*I*(trace(cov)/D)
      prec = inv(cov)
    """
    train_emb = train_emb.astype(np.float64)
    D = train_emb.shape[1]

    mus = np.zeros((K, D), dtype=np.float64)
    for c in range(K):
        Xc = train_emb[train_y == c]
        if len(Xc) == 0:
            raise RuntimeError(f"No train embeddings for class index {c}")
        mus[c] = Xc.mean(axis=0)

    X_centered = train_emb - mus[train_y].astype(np.float64)
    cov = (X_centered.T @ X_centered) / max(1, (X_centered.shape[0] - 1))

    cov = (1 - shrink) * cov + shrink * np.eye(D) * (np.trace(cov) / D)
    prec = np.linalg.inv(cov)

    return mus, prec

def maha_min_dist(X, mus, prec):
    """
    Returns:
      min_d: (N,) min Mahalanobis distance across classes
      argmin: (N,) argmin class index
    """
    X = X.astype(np.float64)
    dists = []
    for c in range(mus.shape[0]):
        diff = X - mus[c]
        d = np.einsum("ni,ij,nj->n", diff, prec, diff)
        dists.append(d)
    dists = np.stack(dists, axis=1)  # (N,K)
    return dists.min(axis=1), dists.argmin(axis=1)

def score_knownness(emb, mus, prec):
    """
    score = -min_dist (higher => more known)
    pred_class = argmin
    """
    min_d, argmin = maha_min_dist(emb, mus, prec)
    score = -min_d
    return score, argmin

def tune_threshold_balanced(val_score, val_pred_class, val_y, K, n_sweep=300):
    """
    Same idea as your notebook:
      Sweep threshold on score, reject to unknown if score < t
      Metric = 0.5*(known_acc + unk_rec)
    """
    lo = np.percentile(val_score, 1)
    hi = np.percentile(val_score, 99)
    TH_SWEEP = np.linspace(lo, hi, n_sweep)

    best = None
    for t in TH_SWEEP:
        val_pred = val_pred_class.copy().astype(int)
        val_pred[val_score < t] = K

        known_mask = (val_y != K)
        unk_mask   = (val_y == K)

        known_acc = (val_pred[known_mask] == val_y[known_mask]).mean() if known_mask.sum() else 0.0
        unk_rec   = (val_pred[unk_mask] == K).mean() if unk_mask.sum() else 0.0

        metric = 0.5 * (known_acc + unk_rec)
        if (best is None) or (metric > best["metric"]):
            best = {"t": float(t), "known_acc": float(known_acc), "unk_rec": float(unk_rec), "metric": float(metric)}

    return best

def compute_val_auroc(val_score, val_y, K):
    """
    Your notebook: AUROC uses (-val_score) as unknownness.
    """
    val_is_unknown = (val_y == K).astype(int)
    return roc_auc_score(val_is_unknown, (-val_score))

def apply_threshold(score, pred_class, t_star, K):
    pred = pred_class.copy().astype(int)
    pred[score < t_star] = K
    return pred
