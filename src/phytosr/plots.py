import os
import numpy as np
import matplotlib.pyplot as plt

def plot_cm_counts(cm, labels, out_path, title, annotate_min=10, figsize=(14,12), dpi=200):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            if v > 0 and (v >= annotate_min or i == j):
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_cm_row_normalized(cm, labels, out_path, title, annotate_threshold=0.10, figsize=(14,12), dpi=200):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    mat = cm / row_sums_safe

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat[i, j])
            if v >= annotate_threshold or (i == j and v > 0):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_known_only_paper(cm_known, known_classes, out_path,
                         title="Known-only Confusion Matrix (Counts)",
                         cmap="YlGnBu", vmin=0, vmax=None,
                         figsize=(11,9), dpi=250,
                         rot_x=60, font_tick=10, font_title=14, font_ann=9,
                         annotate_nonzero_only=True):
    K = len(known_classes)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    im = ax.imshow(cm_known, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=font_title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Output class")

    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(known_classes, rotation=rot_x, ha="right", fontsize=font_tick)
    ax.set_yticklabels(known_classes, fontsize=font_tick)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    vmax_eff = cm_known.max() if vmax is None else vmax
    for i in range(K):
        for j in range(K):
            v = int(cm_known[i, j])
            if annotate_nonzero_only and v == 0:
                continue
            text_color = "white" if vmax_eff > 0 and (v / vmax_eff) > 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=font_ann, color=text_color)

    ax.set_xticks(np.arange(-.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-.5, K, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
