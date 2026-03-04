# scripts/run_clustering_colab.py
import argparse
import os
import sys
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phytosr.datasets import make_loader
from phytosr.methods.mahalanobis import extract_logits_and_embeddings
from phytosr.clustering import cluster_unknowns_from_openmax_logits_gate
from phytosr.colab_cluster_utils import load_openmax_artifacts_newest

def load_sykepic_model(model_dir: str, sykepic_repo: str):
    if sykepic_repo:
        if sykepic_repo not in sys.path:
            sys.path.insert(0, sykepic_repo)
    from sykepic.compute.probability import prepare_model  # type: ignore
    net, classes, img_shape, eval_transform, device = prepare_model(model_dir)
    net = net.to(device).eval()
    return net, list(classes), eval_transform, device

def resolve_model_dir(paths_cfg: dict) -> str:
    if paths_cfg.get("model_dir"):
        return paths_cfg["model_dir"]
    model_out_dir = paths_cfg["model_out_dir"]
    pointer = os.path.join(model_out_dir, "metadata", "latest_model_dir.txt")
    if os.path.exists(pointer):
        return open(pointer, "r", encoding="utf-8").read().strip()
    raise RuntimeError(
        "MODEL_DIR not found. Set paths.model_dir or ensure latest_model_dir.txt exists."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()

    paths_cfg = yaml.safe_load(open(args.paths, "r", encoding="utf-8"))["paths"]
    exp_cfg = yaml.safe_load(open(args.exp, "r", encoding="utf-8"))

    out_base = paths_cfg["out_base"]
    osr_test_root = os.path.join(out_base, "osr_test_root")
    unknown_name = exp_cfg.get("data_prep", {}).get("unknown_name", "__unknown__")

    model_dir = resolve_model_dir(paths_cfg)
    sykepic_repo = paths_cfg.get("sykepic_repo", "")

    bs = int(exp_cfg.get("runtime", {}).get("batch_size", 128))
    nw = int(exp_cfg.get("runtime", {}).get("num_workers", 2))

    cl_cfg = exp_cfg.get("clustering", {})
    om_cfg = exp_cfg.get("openmax", {})

    # ---- Load model ----
    net, known_classes, eval_transform, device = load_sykepic_model(model_dir, sykepic_repo)

    # ---- OSR test inference ----
    _, test_loader = make_loader(
        osr_test_root,
        known_classes,
        eval_transform,
        batch_size=bs,
        num_workers=nw,
        include_unknown=True,
        unknown_name=unknown_name,
    )

    test_logits, test_emb, test_y, test_paths = extract_logits_and_embeddings(net, test_loader, device)

    # ---- Ensure OpenMax artifacts exist (newest detection) ----
    load_openmax_artifacts_newest(model_dir)

    # ---- Run clustering pipeline ----
    info = cluster_unknowns_from_openmax_logits_gate(
        model_dir=model_dir,
        test_logits=test_logits,
        test_emb=test_emb,
        test_paths=test_paths,
        openmax_tailsize=int(om_cfg.get("tailsize", 20)),
        openmax_alpha=int(om_cfg.get("alpha", 3)),
        q_unknown=float(cl_cfg.get("q_unknown", 0.20)),
        min_cluster_size=int(cl_cfg.get("min_cluster_size", 30)),
        min_samples=int(cl_cfg.get("min_samples", 5)),
        umap_neighbors=int(cl_cfg.get("umap_neighbors", 30)),
        umap_min_dist=float(cl_cfg.get("umap_min_dist", 0.1)),
        out_root=os.path.join(model_dir, "clusters"),
        out_prefix=str(cl_cfg.get("out_prefix", "openmax_gate")),
        seed=int(cl_cfg.get("seed", 42)),
    )

    print("Clustering complete.")
    for k, v in info.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    try:
        import umap  # noqa
        import hdbscan  # noqa
    except Exception:
        raise SystemExit(
            "Clustering dependencies missing.\n"
            "In Colab run:\n"
            "  pip install umap-learn hdbscan"
        )

    main()