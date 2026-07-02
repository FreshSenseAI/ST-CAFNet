from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

TARGETS = ["tvc", "tvbn", "tbars"]
REQUIRED_COLUMNS = [
    "sample_id", "image_path", "enose_path", "treatment", "day", *TARGETS
]


def image_transform(image_size: int, training: bool):
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)], p=0.3
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return transforms.Compose(operations)


class FoldScaler:
    def fit(self, frame: pd.DataFrame, enose_root: Path) -> "FoldScaler":
        arrays = [
            load_enose(enose_root / path) for path in frame["enose_path"].tolist()
        ]
        stacked = np.concatenate(arrays, axis=0)
        self.enose_min = stacked.min(axis=0)
        self.enose_max = stacked.max(axis=0)
        labels = frame[TARGETS].to_numpy(np.float32)
        self.label_mean = labels.mean(axis=0)
        self.label_std = labels.std(axis=0)
        self.enose_range = np.maximum(self.enose_max - self.enose_min, 1e-8)
        self.label_std = np.maximum(self.label_std, 1e-8)
        return self

    def transform_enose(self, x: np.ndarray) -> np.ndarray:
        return (x - self.enose_min) / self.enose_range

    def transform_labels(self, y: np.ndarray) -> np.ndarray:
        return (y - self.label_mean) / self.label_std

    def inverse_labels(self, y: np.ndarray) -> np.ndarray:
        return y * self.label_std + self.label_mean

    def state_dict(self) -> dict[str, list[float]]:
        return {
            key: getattr(self, key).tolist()
            for key in ("enose_min", "enose_max", "label_mean", "label_std")
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "FoldScaler":
        obj = cls()
        for key, value in state.items():
            setattr(obj, key, np.asarray(value, dtype=np.float32))
        obj.enose_range = np.maximum(obj.enose_max - obj.enose_min, 1e-8)
        return obj


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique.")
    return frame


def split_manifest(
    frame: pd.DataFrame, test_fraction: float = 0.15, folds: int = 5, seed: int = 42
) -> tuple[pd.DataFrame, list[tuple[pd.DataFrame, pd.DataFrame]]]:
    strata = frame["treatment"].astype(str) + "__day_" + frame["day"].astype(str)
    development, test = train_test_split(
        frame, test_size=test_fraction, random_state=seed, stratify=strata
    )
    dev_strata = (
        development["treatment"].astype(str)
        + "__day_"
        + development["day"].astype(str)
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_frames = []
    for train_idx, val_idx in cv.split(development, dev_strata):
        fold_frames.append(
            (
                development.iloc[train_idx].reset_index(drop=True),
                development.iloc[val_idx].reset_index(drop=True),
            )
        )
    return test.reset_index(drop=True), fold_frames


def load_enose(path: Path) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if array.ndim != 2 or array.shape[1] != 10:
        raise ValueError(f"{path} must have shape [time, 10], got {array.shape}.")
    return array


class SalmonDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        enose_root: str | Path,
        scaler: FoldScaler,
        image_size: int = 224,
        enose_length: int = 120,
        training: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.enose_root = Path(enose_root)
        self.scaler = scaler
        self.enose_length = enose_length
        self.transform = image_transform(image_size, training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        image_path = self.image_root / row.image_path
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        enose = load_enose(self.enose_root / row.enose_path)
        if enose.shape[0] != self.enose_length:
            raise ValueError(
                f"{row.enose_path} has {enose.shape[0]} steps; expected "
                f"{self.enose_length}."
            )
        labels = row[TARGETS].to_numpy(dtype=np.float32)
        return {
            "sample_id": row.sample_id,
            "image": self.transform(image),
            "enose": torch.from_numpy(self.scaler.transform_enose(enose)),
            "targets": torch.from_numpy(self.scaler.transform_labels(labels)),
        }
