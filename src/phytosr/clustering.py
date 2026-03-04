# src/phytosr/clustering.py
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from .openmax_gate import load_openmax_logits_artifacts, openmax_probs_logits_saved


def infer_source_class_from_filename(path: str):
    fn = os.path.basename(path).replace(".png", "")
    return fn.split("__")[0] if "__" in fn else None


def l2_normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def select_topq_tiesafe(scores, q_unknown):
    """
    Tie-safe selection:
      - choose k = ceil(q*N), clamped to [1, N] when N>0
      - tau_eff = score at kth position
      - select all with score >= tau_eff (includes ties)

    NOTE: In the final pipeline below we use EXACT-k selection (to match the original notebook),
    but we keep this helper for optional experimentation.
    """
    N = len(scores)
    if N == 0:
        return np.zeros((0,), dtype=bool), float("nan"), 0

    k = int(np.ceil(q_unknown * N))
    k = max(1, min(k, N))

    idx_sorted = np.argsort(scores)[::-1]
    top_idx = idx_sorted[:k]
    tau_eff = float(scores[top_idx[-1]])
    sel = scores >= tau_eff
    return sel, tau_eff, k


def cluster_unknowns_from_openmax_logits_gate(
    model_dir,
    test_logits,
    test_emb,
    test_paths,
    openmax_tailsize=20,
    openmax_alpha=10,
    q_unknown=0.20,
    min_cluster_size=30,
    min_samples=5,
    umap_neighbors=30,
    umap_min_dist=0.1,
    out_root=None,
    out_prefix="openmax_gate",
    seed=42,
):
    try:
        import hdbscan
        import umap
    except ImportError as e:
        raise ImportError(
            "Clustering dependencies missing. Install in Colab with: pip install umap-learn hdbscan"
        ) from e

    if out_root is None:
        out_root = os.path.join(model_dir, "clusters")
    os.makedirs(out_root, exist_ok=True)

    # ---- Load OpenMax artifacts ----
    mavs, weibulls, alpha_eff, dist_name = load_openmax_logits_artifacts(
        model_dir, tailsize=openmax_tailsize, alpha=openmax_alpha
    )
    test_p = openmax_probs_logits_saved(
        test_logits, mavs, weibulls, alpha=alpha_eff, dist=dist_name
    )
    punk_all = test_p[:, -1]

    # ---- EXACT-k unknown selection (matches your original notebook) ----
    N = len(punk_all)
    k = int(np.ceil(q_unknown * N))
    k = max(1, min(k, N))

    idx_sorted = np.argsort(punk_all)[::-1]  # descending
    top_idx = idx_sorted[:k]

    sel_mask = np.zeros(N, dtype=bool)
    sel_mask[top_idx] = True

    tau_eff = float(punk_all[top_idx[-1]])

    # ---- Select embeddings ----
    X = test_emb[sel_mask].copy()
    # Keep normalization OFF (matches your current "good" Colab behavior).
    # If you want normalized embeddings, uncomment:
    # X = l2_normalize(X)

    sel_paths = np.array(test_paths)[sel_mask].copy()
    sel_scores = punk_all[sel_mask].copy()

    if X.shape[0] < 2:
        raise RuntimeError(
            f"Too few samples selected for clustering (n_selected={X.shape[0]}). "
            f"Increase q_unknown (currently {q_unknown}) or check OpenMax artifacts."
        )

    # ---- HDBSCAN clustering in embedding space ----
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
    )
    clabels = clusterer.fit_predict(X)

    n_clusters = len(set(clabels)) - (1 if -1 in clabels else 0)
    noise_frac = float(np.mean(clabels == -1))

    # ---- UMAP projection (visualization only) ----
    umap_neighbors_eff = min(int(umap_neighbors), max(2, X.shape[0] - 1))

    reducer = umap.UMAP(
        n_neighbors=int(umap_neighbors_eff),
        min_dist=float(umap_min_dist),
        metric="cosine",
        random_state=int(seed),
    )
    X2 = reducer.fit_transform(X)

    # ---- Results table ----
    df = pd.DataFrame(
        {
            "path": sel_paths,
            "score_openmax_punk": sel_scores,
            "cluster": clabels,
            "source_class": [infer_source_class_from_filename(p) for p in sel_paths],
        }
    )

    cluster_summary = []
    for cid in sorted(df["cluster"].unique()):
        if cid == -1:
            continue

        dfc = df[df["cluster"] == cid]
        counts = Counter([c for c in dfc["source_class"].tolist() if c is not None])
        top_class, top_n = counts.most_common(1)[0] if counts else ("unknown", 0)
        frac = top_n / len(dfc) if len(dfc) else 0.0

        cluster_summary.append(
            {
                "cluster": int(cid),
                "n": int(len(dfc)),
                "top_class": top_class,
                "top_class_frac": float(frac),
                "mean_punk": float(dfc["score_openmax_punk"].mean()),
                "median_punk": float(dfc["score_openmax_punk"].median()),
            }
        )

    # Robust if no clusters found (all noise)
    if len(cluster_summary) == 0:
        summary_df = pd.DataFrame(
            columns=["cluster", "n", "top_class", "top_class_frac", "mean_punk", "median_punk"]
        )
    else:
        summary_df = pd.DataFrame(cluster_summary).sort_values(["n"], ascending=False)

    # ---- Save CSVs ----
    out_all = os.path.join(out_root, f"{out_prefix}_all.csv")
    out_sum = os.path.join(out_root, f"{out_prefix}_summary.csv")
    df.to_csv(out_all, index=False)
    summary_df.to_csv(out_sum, index=False)

    # ---- Save UMAP scatter ----
    out_umap = os.path.join(out_root, f"{out_prefix}_umap.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=clabels, s=12)
    plt.title("UMAP of predicted-unknown embeddings (logit-OpenMax gate → HDBSCAN)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_umap, dpi=300)
    plt.close()

    # ---- Montage strips ----
    def load_rgb(path):
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def show_strip(ax, paths, k_strip=10, size=70):
        paths = list(paths)
        if len(paths) == 0:
            ax.axis("off")
            return
        choose = paths if len(paths) <= k_strip else random.sample(paths, k_strip)
        canvas = np.ones((size, k_strip * size, 3), dtype=np.uint8) * 255
        for i, p in enumerate(choose):
            img = load_rgb(p)
            if img is None:
                continue
            img = cv2.resize(img, (size, size))
            canvas[:, i * size : (i + 1) * size] = img
        ax.imshow(canvas)
        ax.axis("off")

    def dominant_label_and_pct(dfc):
        sc = dfc["source_class"].dropna()
        if len(sc) == 0:
            return ("unknown", 0)
        vc = sc.value_counts()
        top_label = vc.index[0]
        top_n = int(vc.iloc[0])
        pct = int(round(100.0 * top_n / len(dfc)))
        return (top_label, pct)

    show_top_n = 6
    clusters_to_show = (
        df[df["cluster"] != -1]["cluster"].value_counts().head(show_top_n).index.tolist()
    )

    nrows = 1 + len(clusters_to_show)
    fig = plt.figure(figsize=(14, 3.5 + 2.4 * len(clusters_to_show)))

    ax0 = fig.add_subplot(nrows, 1, 1)
    ax0.scatter(X2[:, 0], X2[:, 1], c=clabels, s=10)
    ax0.set_title("UMAP projection of unknown candidates (colored by HDBSCAN cluster)")
    ax0.set_xlabel("UMAP-1")
    ax0.set_ylabel("UMAP-2")

    for row_i, cid in enumerate(clusters_to_show, start=2):
        ax = fig.add_subplot(nrows, 1, row_i)
        dfc = df[df["cluster"] == cid].copy()
        dfc = dfc.sort_values("score_openmax_punk", ascending=False)
        top_label, top_pct = dominant_label_and_pct(dfc)
        ax.set_title(f"Cluster {cid} — {top_label} ({top_pct}%)  n={len(dfc)}", loc="left")
        show_strip(ax, dfc["path"].tolist(), k_strip=10, size=70)

    plt.tight_layout()
    out_strips = os.path.join(out_root, f"{out_prefix}_umap_plus_cluster_strips_with_pct.png")
    plt.savefig(out_strips, dpi=300)
    plt.close()

    info = {
        "model_dir": model_dir,
        "out_root": out_root,
        "out_prefix": out_prefix,
        "q_unknown": float(q_unknown),
        "k": int(k),
        "tau_eff": float(tau_eff),
        "n_selected": int(sel_mask.sum()),
        "n_total": int(len(punk_all)),
        "n_clusters": int(n_clusters),
        "noise_frac": float(noise_frac),
        "openmax_alpha_used": int(alpha_eff),
        "openmax_dist": str(dist_name),
        "csv_all": out_all,
        "csv_summary": out_sum,
        "umap_png": out_umap,
        "montage_png": out_strips,
    }
    return info