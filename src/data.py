"""Data preparation for the chest X-ray pneumonia dataset.

Handles the two fixes this project needed on top of the raw Kaggle dataset:
1. The provided `val` split has only 16 images, too small for a reliable signal.
2. Filenames encode a patient/study id (e.g. `person1003_bacteria_2934.jpeg`),
   and a naive image-level split leaks the same patient across train/val.

`build_patient_split` re-splits `train` into `new_train`/`new_val` at the
patient level so no patient's images end up on both sides.
"""

import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose

CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def patient_id(filename: str, cls: str) -> str:
    """Extracts a patient/study identifier from an image filename.

    PNEUMONIA files look like `person1003_bacteria_2934.jpeg` -> `person1003`.
    NORMAL files look like `IM-0115-0001.jpeg` -> `IM-0115`.
    Falls back to the full filename if the pattern doesn't match, so an
    unrecognized file is treated as its own single-image "patient" rather
    than silently grouped with something else.
    """
    if cls == "PNEUMONIA":
        match = re.match(r"(person\d+)_", filename)
    else:
        match = re.match(r"(IM-\d+)-", filename)
    return match.group(1) if match else filename


def build_patient_split(
    train_dir: Path,
    new_train_dir: Path,
    new_val_dir: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    class_names: tuple = CLASS_NAMES,
    force: bool = False,
) -> None:
    """Re-splits `train_dir` into `new_train_dir`/`new_val_dir`, grouped by patient.

    Idempotent: if `new_train_dir` already exists, does nothing unless `force=True`,
    since copying thousands of images is slow and only needs to happen once.
    """
    if new_train_dir.exists() and not force:
        print(f"{new_train_dir} already exists, skipping split (pass force=True to rebuild).")
        return

    # Clear any stale split before rebuilding, otherwise old and new copies mix
    # together and reintroduce patient leakage between runs.
    if new_train_dir.exists():
        shutil.rmtree(new_train_dir)
    if new_val_dir.exists():
        shutil.rmtree(new_val_dir)

    random.seed(seed)

    for cls in class_names:
        src_folder = train_dir / cls
        images = sorted(src_folder.glob("*.jpeg"))  # sorted: glob() order isn't guaranteed stable

        by_patient = defaultdict(list)
        for img in images:
            by_patient[patient_id(img.name, cls)].append(img)

        patients = sorted(by_patient.keys())
        random.shuffle(patients)

        n_val_patients = int(len(patients) * val_fraction)
        val_patients = set(patients[:n_val_patients])

        val_images = [img for pid in val_patients for img in by_patient[pid]]
        train_images = [
            img for pid in patients if pid not in val_patients for img in by_patient[pid]
        ]

        (new_val_dir / cls).mkdir(parents=True, exist_ok=True)
        (new_train_dir / cls).mkdir(parents=True, exist_ok=True)

        for img in val_images:
            shutil.copy(img, new_val_dir / cls / img.name)
        for img in train_images:
            shutil.copy(img, new_train_dir / cls / img.name)

        print(
            f"{cls} | patients train: {len(patients) - len(val_patients)} "
            f"val: {len(val_patients)} | images train: {len(train_images)} val: {len(val_images)}"
        )


def assert_no_patient_leakage(new_train_dir: Path, new_val_dir: Path, class_names: tuple = CLASS_NAMES) -> None:
    """Raises if any patient's images ended up on both sides of the split."""
    for cls in class_names:
        train_ids = {patient_id(p.name, cls) for p in (new_train_dir / cls).glob("*.jpeg")}
        val_ids = {patient_id(p.name, cls) for p in (new_val_dir / cls).glob("*.jpeg")}
        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(f"{cls}: {len(overlap)} patients leaked between train and val: {overlap}")


def get_dataloaders(
    new_train_dir: Path,
    new_val_dir: Path,
    test_dir: Path,
    transform: Compose,
    batch_size: int = 32,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train/val/test DataLoaders from the three image folders."""
    import os

    if num_workers is None:
        num_workers = os.cpu_count()

    train_data = datasets.ImageFolder(root=new_train_dir, transform=transform)
    val_data = datasets.ImageFolder(root=new_val_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader
