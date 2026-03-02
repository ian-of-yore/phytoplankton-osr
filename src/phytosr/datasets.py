import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

UNKNOWN_NAME_DEFAULT = "__unknown__"

class CV2FolderOSR(Dataset):
    """
    ImageFolder-like dataset:
      root/
        ClassA/*.png
        ClassB/*.png
        __unknown__/*.png   (optional; include_unknown=True)
    Uses cv2 -> RGB -> provided transform.
    """
    def __init__(self, root, known_class_order, transform, include_unknown=False, unknown_name=UNKNOWN_NAME_DEFAULT):
        self.root = root
        self.transform = transform
        self.known_order = list(known_class_order)
        self.class_to_idx = {c: i for i, c in enumerate(self.known_order)}
        self.include_unknown = include_unknown
        self.unknown_name = unknown_name
        self.K = len(self.known_order)

        self.samples = []
        self.classes_present = []

        for cname in sorted(os.listdir(root)):
            cdir = os.path.join(root, cname)
            if not os.path.isdir(cdir):
                continue

            if cname == self.unknown_name:
                if not include_unknown:
                    continue
                y = self.K
            else:
                if cname not in self.class_to_idx:
                    # ignore classes not in model class list
                    continue
                y = self.class_to_idx[cname]

            self.classes_present.append(cname)
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(cdir, fn), y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img) if self.transform else img
        return x, y, path


def make_loader(root, known_classes, transform, batch_size=128, num_workers=2, include_unknown=False, unknown_name="__unknown__"):
    ds = CV2FolderOSR(root, known_classes, transform, include_unknown=include_unknown, unknown_name=unknown_name)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds, dl


@torch.no_grad()
def extract_logits(model, loader, device):
    logits_all, y_all, paths_all = [], [], []
    for x, y, paths in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(y.numpy())
        paths_all.extend(list(paths))
    import numpy as np
    return np.concatenate(logits_all), np.concatenate(y_all), paths_all
