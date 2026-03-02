import os
import pickle
import numpy as np
from scipy.stats import weibull_min

def load_openmax_logits_artifacts(model_dir, tailsize=None, alpha=None):
    """
    Supports:
      - our repo artifacts: openmax_mavs_logits_t{tailsize}_a{alpha}.npy + openmax_weibull_logits_t{tailsize}_a{alpha}.pkl
      - legacy names: openmax_mavs_logits_best.npy + openmax_weibull_logits_best.pkl
    Returns: (mavs, weibulls, alpha, dist_name)
    """
    # 1) Try configured naming
    if (tailsize is not None) and (alpha is not None):
        mav_path = os.path.join(model_dir, f"openmax_mavs_logits_t{tailsize}_a{alpha}.npy")
        wb_path  = os.path.join(model_dir, f"openmax_weibull_logits_t{tailsize}_a{alpha}.pkl")
        if os.path.exists(mav_path) and os.path.exists(wb_path):
            mavs = np.load(mav_path)
            with open(wb_path, "rb") as f:
                wb = pickle.load(f)
            weibulls = np.array(wb["weibulls"], dtype=np.float64)
            # repo pickle stores "euclidean": bool
            dist_name = "euclidean" if bool(wb.get("euclidean", True)) else "cosine"
            alpha_eff = int(wb.get("alpha", alpha))
            return mavs, weibulls, alpha_eff, dist_name

    # 2) Try legacy "best" naming
    mav_path = os.path.join(model_dir, "openmax_mavs_logits_best.npy")
    wb_path  = os.path.join(model_dir, "openmax_weibull_logits_best.pkl")
    if os.path.exists(mav_path) and os.path.exists(wb_path):
        mavs = np.load(mav_path)
        with open(wb_path, "rb") as f:
            wb = pickle.load(f)
        weibulls = np.array(wb["weibulls"], dtype=np.float64)
        alpha_eff = int(wb.get("alpha", alpha if alpha is not None else 10))
        # legacy might store "dist" string
        dist_name = wb.get("dist", "euclidean")
        return mavs, weibulls, alpha_eff, dist_name

    raise FileNotFoundError(
        "Could not find OpenMax logit artifacts. "
        "Run OpenMax first, or place artifacts in MODEL_DIR."
    )

def openmax_probs_logits_saved(logits, mavs, weibulls, alpha, dist="euclidean"):
    """
    Same as your notebook Block 7.2.
    Returns (N, K+1) probabilities, last column is P_unknown.
    """
    N, K = logits.shape
    out = np.zeros((N, K+1), dtype=np.float32)
    use_euclid = str(dist).lower().startswith("euc")

    for i in range(N):
        x = logits[i].astype(np.float64)
        top_idx = np.argsort(x)[::-1][:alpha]

        revised = x.copy()
        unknown_mass = 0.0

        for c in top_idx:
            shape, scale = weibulls[c]

            if use_euclid:
                d = float(np.linalg.norm(x - mavs[c]))
            else:
                xn = x / (np.linalg.norm(x) + 1e-12)
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
