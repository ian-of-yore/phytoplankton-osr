import os
import subprocess
import glob


def render_ini(template_path, output_path, replacements: dict):
    with open(template_path, "r") as f:
        txt = f.read()
    for k, v in replacements.items():
        txt = txt.replace("{" + k + "}", str(v))
    with open(output_path, "w") as f:
        f.write(txt)


def run_training(ini_path):
    cmd = ["sykepic", "train", ini_path]
    subprocess.run(cmd, check=True)


def find_latest_model(model_out_dir, prefix="resnet18_"):
    candidates = sorted(
        glob.glob(os.path.join(model_out_dir, prefix + "*")),
        key=os.path.getmtime
    )
    if not candidates:
        raise RuntimeError("No trained models found.")
    return candidates[-1]
