import os
import subprocess
import glob
import sys


def render_ini(template_path, output_path, replacements: dict):
    """
    Render a sykepic .ini template by replacing {placeholders} with values.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        txt = f.read()
    for k, v in replacements.items():
        txt = txt.replace("{" + k + "}", str(v))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(txt)


def run_training(ini_path: str):
    """
    Run sykepic training via a local shim that patches label dtype issues on some setups.
    This avoids crashes like: 'expected scalar type Long but found Int'.
    """
    cmd = [sys.executable, "-m", "phytosr.sykepic_train_shim", ini_path]
    subprocess.run(cmd, check=True)


def find_latest_model(model_out_dir: str, prefix: str = "resnet18_") -> str:
    """
    Find newest model directory matching prefix (e.g., resnet18_1, resnet18_2, ...).
    """
    candidates = sorted(
        glob.glob(os.path.join(model_out_dir, prefix + "*")),
        key=os.path.getmtime
    )
    if not candidates:
        raise RuntimeError(f"No trained models found in: {model_out_dir}")
    return candidates[-1]