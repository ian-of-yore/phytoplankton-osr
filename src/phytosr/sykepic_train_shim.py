import sys
from typing import Any


def _patch_torch_label_dtype() -> None:
    """
    Patch torch cross-entropy to force target labels to int64 (Long),
    required by CrossEntropyLoss on some setups.
    """
    import torch
    import torch.nn.functional as F

    if getattr(F.cross_entropy, "_phytosr_patched", False):
        return

    _orig = F.cross_entropy

    def _wrapped(input: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any):
        if isinstance(target, torch.Tensor) and target.dtype != torch.long:
            target = target.long()
        return _orig(input, target, *args, **kwargs)

    _wrapped._phytosr_patched = True  # type: ignore[attr-defined]
    F.cross_entropy = _wrapped  # type: ignore[assignment]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m phytosr.sykepic_train_shim <path_to_train_ini>")

    ini_path = sys.argv[1]

    # Patch BEFORE syke-pic training starts
    _patch_torch_label_dtype()

    # Recreate argv as syke-pic expects for your version: positional ini path
    sys.argv = ["sykepic", "train", ini_path]

    from sykepic.__main__ import main as sykepic_main
    sykepic_main()


if __name__ == "__main__":
    main()