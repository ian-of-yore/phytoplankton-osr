import argparse
import yaml
import os
import sys
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phytosr.sykepic_engine import render_ini, run_training, find_latest_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_base = cfg["paths"]["out_base"]
    model_out_dir = cfg["paths"]["model_out_dir"]

    train_root = os.path.join(out_base, "train_root")
    sykepic_test_root = os.path.join(out_base, "sykepic_test_root")

    template_ini = os.path.join(REPO_ROOT, cfg["sykepic"]["train_ini_template"])
    rendered_ini = os.path.join(model_out_dir, "train_rendered.ini")

    os.makedirs(model_out_dir, exist_ok=True)

    render_ini(
        template_ini,
        rendered_ini,
        {
            "train_root": train_root,
            "sykepic_test_root": sykepic_test_root,
            "model_out_dir": model_out_dir,
        },
    )

    print("Rendered ini:", rendered_ini)

    # Auto-fix GPU flag depending on runtime
    try:
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False

    if not has_cuda:
        # Make sure SYKEPIC doesn't try CUDA on CPU-only PyTorch builds
        with open(rendered_ini, "r", encoding="utf-8") as f:
            ini_txt = f.read()
        ini_txt = ini_txt.replace("gpu = yes", "gpu = no")
        with open(rendered_ini, "w", encoding="utf-8") as f:
            f.write(ini_txt)
        print("⚠️ CUDA not available: forcing gpu = no in rendered ini")

    run_training(rendered_ini)

    latest_model = find_latest_model(
        model_out_dir,
        prefix=cfg["sykepic"].get("network_prefix", "resnet18_"),
    )

    meta_dir = os.path.join(model_out_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    pointer_path = os.path.join(meta_dir, "latest_model_dir.txt")
    with open(pointer_path, "w") as f:
        f.write(latest_model)

    print("\n✅ Training complete.")
    print("Latest model:", latest_model)
    print("Pointer saved at:", pointer_path)


if __name__ == "__main__":
    main()
