import argparse
import yaml
import os
import sys

# Allow running without installing as a package
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phytosr.data_prep import SplitConfig, prepare_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Experiment config (e.g., configs/exp.yaml)")
    ap.add_argument("--paths", required=True, help="Machine-specific paths (e.g., configs/paths.yaml)")
    args = ap.parse_args()

    # Load experiment config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_y = yaml.safe_load(f)

    # Load paths config and override cfg_y["paths"]
    with open(args.paths, "r", encoding="utf-8") as f:
        paths_cfg = yaml.safe_load(f)
    cfg_y["paths"] = paths_cfg["paths"]

    d = cfg_y["data_prep"]

    # Paths (machine-specific)
    out_base = cfg_y["paths"]["out_base"]
    dataset_root = cfg_y["paths"]["dataset_root"]

    cfg = SplitConfig(
        dataset_root=dataset_root,
        out_base=out_base,
        min_count=int(d.get("min_count", 50)),
        seed=int(d.get("seed", 42)),
        known_train_frac=float(d.get("known_train_frac", 0.70)),
        known_val_frac=float(d.get("known_val_frac", 0.15)),
        known_test_frac=float(d.get("known_test_frac", 0.15)),
        unk_val_frac=float(d.get("unk_val_frac", 0.60)),
        unk_test_frac=float(d.get("unk_test_frac", 0.40)),
        img_ext=str(d.get("img_ext", ".png")),
        unknown_name=str(d.get("unknown_name", "__unknown__")),
        use_symlinks=bool(d.get("use_symlinks", False)),
        delete_out_base_if_exists=bool(d.get("delete_out_base_if_exists", False)),
        make_zip=bool(d.get("make_zip", False)),
        zip_name=str(d.get("zip_name", "dataset_splits_all4folders")),
    )

    out = prepare_splits(cfg)

    print("\n✅ Prepared splits:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()