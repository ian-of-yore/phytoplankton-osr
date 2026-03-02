import os, glob, shutil, random, csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

IMG_EXT_DEFAULT = ".png"
UNKNOWN_NAME_DEFAULT = "__unknown__"


@dataclass
class SplitConfig:
    dataset_root: str
    out_base: str
    min_count: int = 50
    seed: int = 42

    # Known split fractions (train/val/test)
    known_train_frac: float = 0.70
    known_val_frac: float = 0.15
    known_test_frac: float = 0.15

    # Unknown split fractions (val/test)
    unk_val_frac: float = 0.60
    unk_test_frac: float = 0.40

    img_ext: str = IMG_EXT_DEFAULT
    unknown_name: str = UNKNOWN_NAME_DEFAULT

    use_symlinks: bool = False
    delete_out_base_if_exists: bool = False

    # Optional zip (mostly for Colab convenience)
    make_zip: bool = False
    zip_name: str = "dataset_splits_all4folders"


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def nuke_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def list_class_images(root: str, img_ext: str) -> Dict[str, List[str]]:
    classes = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    out: Dict[str, List[str]] = {}
    for cdir in classes:
        cname = os.path.basename(cdir)
        imgs = [p for p in glob.glob(os.path.join(cdir, "**", f"*{img_ext}"), recursive=True)
                if os.path.isfile(p)]
        out[cname] = sorted(imgs)
    return out


def link_or_copy(src: str, dst: str, use_symlinks: bool) -> None:
    safe_mkdir(os.path.dirname(dst))
    if use_symlinks:
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _split_counts(n: int, frac_b: float, frac_c: float) -> Tuple[int, int, int]:
    """Return (a,b,c) where b,c are rounded and a is remainder."""
    b = int(round(n * frac_b))
    c = int(round(n * frac_c))
    a = n - b - c
    if a < 0:
        overflow = -a
        take = min(overflow, b); b -= take; overflow -= take
        take = min(overflow, c); c -= take; overflow -= take
        a = n - b - c
    return a, b, c


def split_known(paths: List[str], seed: int, val_frac: float, test_frac: float) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)

    n_train, n_val, n_test = _split_counts(n, val_frac, test_frac)

    # Keep at least 1 val/test when possible
    if n >= 3:
        n_val = max(n_val, 1)
        n_test = max(n_test, 1)
        n_train = n - n_val - n_test

    return paths[:n_train], paths[n_train:n_train+n_val], paths[n_train+n_val:]


def split_unknown(paths: List[str], seed: int, val_frac: float) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)

    if n == 1:
        return [], paths
    if n == 2:
        return [paths[0]], [paths[1]]

    n_val = int(round(n * val_frac))
    n_val = max(1, min(n_val, n - 1))
    return paths[:n_val], paths[n_val:]


def count_files(root: str, img_ext: str) -> int:
    return len(glob.glob(os.path.join(root, "**", f"*{img_ext}"), recursive=True))


def _write_list(path: str, items: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(str(it).strip() + "\n")


def _write_csv(path: str, header: List[str], rows: List[List]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def prepare_splits(cfg: SplitConfig) -> Dict[str, str]:
    # Validate fractions
    if abs((cfg.known_train_frac + cfg.known_val_frac + cfg.known_test_frac) - 1.0) > 1e-6:
        raise ValueError("Known split fractions must sum to 1.0")
    if abs((cfg.unk_val_frac + cfg.unk_test_frac) - 1.0) > 1e-6:
        raise ValueError("Unknown split fractions must sum to 1.0")

    if cfg.delete_out_base_if_exists:
        nuke_dir(cfg.out_base)
    safe_mkdir(cfg.out_base)

    train_root = os.path.join(cfg.out_base, "train_root")
    osr_val_root = os.path.join(cfg.out_base, "osr_val_root")
    osr_test_root = os.path.join(cfg.out_base, "osr_test_root")
    sykepic_test_root = os.path.join(cfg.out_base, "sykepic_test_root")

    for p in [train_root, osr_val_root, osr_test_root, sykepic_test_root]:
        nuke_dir(p)
        safe_mkdir(p)

    safe_mkdir(os.path.join(osr_val_root, cfg.unknown_name))
    safe_mkdir(os.path.join(osr_test_root, cfg.unknown_name))

    class_to_imgs = list_class_images(cfg.dataset_root, cfg.img_ext)
    class_counts = {c: len(v) for c, v in class_to_imgs.items()}

    known_classes = sorted([c for c, n in class_counts.items() if n >= cfg.min_count])
    unknown_classes = sorted([c for c, n in class_counts.items() if n < cfg.min_count])

    print("OUT_BASE:", cfg.out_base)
    print("DATASET_ROOT:", cfg.dataset_root)
    print("MIN_COUNT:", cfg.min_count)
    print("Total classes:", len(class_to_imgs))
    print("Known classes (>=MIN_COUNT):", len(known_classes))
    print("Unknown classes (<MIN_COUNT):", len(unknown_classes))
    print("Total images:", sum(class_counts.values()))
    print("Known images:", sum(class_counts[c] for c in known_classes))
    print("Unknown images:", sum(class_counts[c] for c in unknown_classes))

    used = set()

    def add_files_unique(file_list: List[str], out_dir: str, class_name: str) -> None:
        safe_mkdir(os.path.join(out_dir, class_name))
        for src in file_list:
            if src in used:
                raise RuntimeError(f"Leakage detected: {src}")
            used.add(src)
            dst = os.path.join(out_dir, class_name, os.path.basename(src))
            link_or_copy(src, dst, cfg.use_symlinks)

    def add_unknown_files_unique(file_list: List[str], out_unknown_dir: str) -> None:
        for src in file_list:
            if src in used:
                raise RuntimeError(f"Leakage detected: {src}")
            used.add(src)
            src_class = os.path.basename(os.path.dirname(src))
            dst_name = f"{src_class}__{os.path.basename(src)}"
            dst = os.path.join(out_unknown_dir, dst_name)
            link_or_copy(src, dst, cfg.use_symlinks)

    per_class_rows = []
    for i, cls in enumerate(known_classes):
        paths = class_to_imgs[cls]
        tr, va, te = split_known(
            paths,
            seed=cfg.seed + i * 1000,
            val_frac=cfg.known_val_frac,
            test_frac=cfg.known_test_frac
        )
        add_files_unique(tr, train_root, cls)
        add_files_unique(va, osr_val_root, cls)
        add_files_unique(te, osr_test_root, cls)
        per_class_rows.append([cls, len(paths), len(tr), len(va), len(te)])

    unk_val_all, unk_test_all = 0, 0
    for j, cls in enumerate(unknown_classes):
        paths = class_to_imgs[cls]
        va_u, te_u = split_unknown(paths, seed=cfg.seed + 999999 + j * 1000, val_frac=cfg.unk_val_frac)
        add_unknown_files_unique(va_u, os.path.join(osr_val_root, cfg.unknown_name))
        add_unknown_files_unique(te_u, os.path.join(osr_test_root, cfg.unknown_name))
        unk_val_all += len(va_u)
        unk_test_all += len(te_u)

    # Build sykepic_test_root: known-only physical copy from osr_test_root
    nuke_dir(sykepic_test_root)
    safe_mkdir(sykepic_test_root)
    for d in glob.glob(os.path.join(osr_test_root, "*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if name == cfg.unknown_name:
            continue
        shutil.copytree(d, os.path.join(sykepic_test_root, name), symlinks=False)

    # Sanity counts
    train_total = count_files(train_root, cfg.img_ext)
    osr_val_total = count_files(osr_val_root, cfg.img_ext)
    osr_val_unknown = count_files(os.path.join(osr_val_root, cfg.unknown_name), cfg.img_ext)
    osr_val_known = osr_val_total - osr_val_unknown

    osr_test_total = count_files(osr_test_root, cfg.img_ext)
    osr_test_unknown = count_files(os.path.join(osr_test_root, cfg.unknown_name), cfg.img_ext)
    osr_test_known = osr_test_total - osr_test_unknown

    sykepic_test_total = count_files(sykepic_test_root, cfg.img_ext)

    print("\nFolder counts:")
    print(" Train total:", train_total)
    print(" OSR val total:", osr_val_total, "| known:", osr_val_known, "| unknown:", osr_val_unknown)
    print(" OSR test total:", osr_test_total, "| known:", osr_test_known, "| unknown:", osr_test_unknown)
    print(" Sykepic test total:", sykepic_test_total)
    print("\nUnknown pooled counts -> val:", unk_val_all, "test:", unk_test_all)

    # Write metadata
    meta_dir = os.path.join(cfg.out_base, "metadata")
    safe_mkdir(meta_dir)

    _write_list(os.path.join(meta_dir, "known_classes.txt"), known_classes)
    _write_list(os.path.join(meta_dir, "unknown_classes.txt"), unknown_classes)

    _write_csv(
        os.path.join(meta_dir, "class_counts.csv"),
        ["class", "count"],
        [[c, class_counts[c]] for c in sorted(class_counts.keys())]
    )

    _write_csv(
        os.path.join(meta_dir, "known_split_per_class.csv"),
        ["class", "total", "train", "osr_val", "osr_test"],
        per_class_rows
    )

    _write_csv(
        os.path.join(meta_dir, "split_details.csv"),
        ["split", "total_images", "known_images", "unknown_images"],
        [
            ["training_dataset", train_total, train_total, 0],
            ["osr_val_dataset", osr_val_total, osr_val_known, osr_val_unknown],
            ["osr_test_dataset", osr_test_total, osr_test_known, osr_test_unknown],
            ["sykepic_test_dataset", sykepic_test_total, sykepic_test_total, 0],
        ]
    )

    # Optional zip (useful in Colab sometimes)
    if cfg.make_zip:
        zip_path = os.path.join(cfg.out_base, f"{cfg.zip_name}.zip")
        tmp_dir = os.path.join("/tmp", f"tmp_{cfg.zip_name}")

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        for folder in ["train_root", "osr_val_root", "osr_test_root", "sykepic_test_root", "metadata"]:
            src = os.path.join(cfg.out_base, folder)
            dst = os.path.join(tmp_dir, folder)
            shutil.copytree(src, dst, symlinks=False)

        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(zip_path[:-4], "zip", tmp_dir)
        print("\nZIP created at:", zip_path)

    return {
        "train_root": train_root,
        "osr_val_root": osr_val_root,
        "osr_test_root": osr_test_root,
        "sykepic_test_root": sykepic_test_root,
        "metadata_dir": meta_dir,
    }
