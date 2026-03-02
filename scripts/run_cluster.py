import argparse
import os
import sys
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phytosr.datasets import make_loader
from phytosr.methods.mahalanobis import extract_logits_and_embeddings  # base[8] hook (reused)
from phytosr.clustering import cluster_unknowns_from_openmax_logits_gate

def load_sykepic_model(model_dir: str, sykepic_repo: str):
    if sykepic_repo not in sys.path:
        sys.path.insert(0, sykepic_repo)
    from sykepic.compute.probability import prepare_model
    net, classes, img_shape, eval_transform, device = prepare_model(model_dir)
    net = net.to(device).eval()
    return net, list(classes), eval_transform, device

def resolve_model_dir(cfg):
    if cfg["paths"].get("model_dir"):
        return cfg["paths"]["model_dir"]
    model_out_dir = cfg["paths"]["model_out_dir"]
    pointer = os.path.join(model_out_dir, "metadata", "latest_model_dir.txt")
    if os.path.exists(pointer):
        return open(pointer, "r", encoding="utf-8").read().strip()
    raise RuntimeError("MODEL_DIR not found. Set paths.model_dir or ensure latest_model_dir.txt exists.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    out_base = cfg["paths"]["out_base"]
    osr_test_root = os.path.join(out_base, "osr_test_root")
    unknown_name = cfg.get("data_prep", {}).get("unknown_name", "__unknown__")

    model_dir = resolve_model_dir(cfg)
    sykepic_repo = cfg["paths"]["sykepic_repo"]

    net, known_classes, eval_transform, device = load_sykepic_model(model_dir, sykepic_repo)

    bs = int(cfg.get("runtime", {}).get("batch_size", 128))
    nw = int(cfg.get("runtime", {}).get("num_workers", 2))

    # Only OSR_TEST is needed for the final clustering pipeline
    test_ds, test_loader = make_loader(
        osr_test_root, known_classes, eval_transform,
        batch_size=bs, num_workers=nw,
        include_unknown=True, unknown_name=unknown_name
    )

    # Extract logits + embeddings from OSR_TEST
    test_logits, test_emb, test_y, test_paths = extract_logits_and_embeddings(net, test_loader, device)

    cl_cfg = cfg.get("clustering", {})
    info = cluster_unknowns_from_openmax_logits_gate(
        model_dir=model_dir,
        test_logits=test_logits,
        test_emb=test_emb,
        test_paths=test_paths,
        openmax_tailsize=int(cfg["openmax"].get("tailsize", 20)),
        openmax_alpha=int(cfg["openmax"].get("alpha", 10)),
        q_unknown=float(cl_cfg.get("q_unknown", 0.20)),
        min_cluster_size=int(cl_cfg.get("min_cluster_size", 30)),
        min_samples=int(cl_cfg.get("min_samples", 5)),
        umap_neighbors=int(cl_cfg.get("umap_neighbors", 30)),
        umap_min_dist=float(cl_cfg.get("umap_min_dist", 0.1)),
        out_root=os.path.join(model_dir, "clusters"),
        out_prefix=str(cl_cfg.get("out_prefix", "openmax_gate")),
        seed=int(cl_cfg.get("seed", 42)),
    )

    print("\n✅ Clustering complete.")
    for k, v in info.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
