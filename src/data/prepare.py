"""Build canonical ID and held-out-event manifests from the official split."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .schema import normalize_event, normalize_label


def _find_image_root(data_root: Path) -> Path:
    candidates = sorted(path for path in data_root.rglob("data_image") if path.is_dir())
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one data_image directory, found {candidates}")
    return candidates[0]


def load_official_agreed(data_root: Path) -> pd.DataFrame:
    split_root = data_root / "splits" / "agreed_label" / "crisismmd_datasplit_agreed_label"
    image_root = _find_image_root(data_root)
    frames = []
    for official, canonical in (("train", "train"), ("dev", "val"), ("test", "test")):
        source = split_root / f"task_informative_text_img_agreed_lab_{official}.tsv"
        frame = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
        frame["official_split"] = official
        frame["split"] = canonical
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    if not (raw.label_text == raw.label_image).all():
        raise ValueError("official agreed split contains text/image label disagreement")
    if not raw.label_text_image.str.lower().eq("positive").all():
        raise ValueError("official agreed split contains a non-positive agreement flag")
    normalized_text = raw.label_text.map(normalize_label)
    normalized_image = raw.label_image.map(normalize_label)
    normalized_target = raw.label.map(normalize_label)
    if not normalized_text.eq(normalized_image).all() or not normalized_target.eq(normalized_text).all():
        raise ValueError("official agreed split target differs from modality labels")
    result = pd.DataFrame({
        "sample_id": raw.image_id.astype(str),
        "tweet_id": raw.tweet_id.astype(str),
        "image_id": raw.image_id.astype(str),
        "event": raw.event_name.map(normalize_event),
        "text": raw.tweet_text,
        "image_path": raw.image.map(lambda value: str((image_root.parent / value).resolve())),
        "text_label": normalized_text,
        "image_label": normalized_image,
        "label": normalized_target,
        "official_split": raw.official_split,
        "protocol": "id_official",
        "split": raw.split,
        "group_id": raw.tweet_id.astype(str),
    })
    if result.sample_id.duplicated().any():
        raise ValueError("duplicate image/sample IDs in official agreed split")
    return result


def build_ood(id_frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    result = id_frame.copy()
    result["protocol"] = "ood_heldout"
    california = result.event.eq("california_wildfires")
    result.loc[california, "split"] = "test"
    candidates = result.loc[~california].copy()
    strata = candidates.event.astype(str) + "::" + candidates.label.astype(str)
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = list(splitter.split(candidates, strata, groups=candidates.group_id))
    target = 0.15 * len(candidates)
    _, val_indices = min(folds, key=lambda pair: abs(len(pair[1]) - target))
    result.loc[~california, "split"] = "train"
    result.loc[candidates.iloc[val_indices].index, "split"] = "val"
    result["official_split"] = id_frame.official_split
    return result


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
