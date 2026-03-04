# src/phytosr/colab_cluster_utils.py
import os
import pickle
import numpy as np

def newest_file(model_dir: str, pattern_substr: str):
    candidates = []
    for fn in os.listdir(model_dir):
        if pattern_substr in fn:
            path = os.path.join(model_dir, fn)
            if os.path.isfile(path):
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def load_openmax_artifacts_newest(model_dir: str):
    """
    Load newest OpenMax MAV + Weibull artifacts from MODEL_DIR.
    Returns:
        mavs, weibulls, alpha, dist_name
    """
    mav_path = newest_file(model_dir, "openmax_mavs_logits")
    wb_path  = newest_file(model_dir, "openmax_weibull_logits")

    if mav_path is None or wb_path is None:
        raise FileNotFoundError(
            f"Missing OpenMax artifacts in MODEL_DIR={model_dir}."
        )

    mavs = np.load(mav_path)
    with open(wb_path, "rb") as f:
        wb = pickle.load(f)

    weibulls = np.array(wb["weibulls"], dtype=np.float64)

    if "euclidean" in wb:
        dist_name = "euclidean" if bool(wb.get("euclidean", True)) else "cosine"
    else:
        dist_name = str(wb.get("dist", "euclidean"))

    alpha = int(wb.get("alpha", 10))

    return mavs, weibulls, alpha, dist_name