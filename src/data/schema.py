"""Canonical CrisisMMD schema and strict binary-label normalization."""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_COLUMNS = (
    "sample_id", "tweet_id", "image_id", "event", "text", "image_path",
    "text_label", "image_label", "label", "official_split", "protocol",
    "split", "group_id",
)

LABEL_MAP = {"not_informative": 0, "informative": 1}


def normalize_label(value: object) -> int:
    key = str(value).strip().lower().replace(" ", "_")
    if key not in LABEL_MAP:
        raise ValueError(f"unknown informativeness label: {value!r}")
    return LABEL_MAP[key]


def normalize_event(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


@dataclass(frozen=True)
class CanonicalRow:
    sample_id: str
    tweet_id: str
    image_id: str
    event: str
    text: str
    image_path: str
    text_label: int
    image_label: int
    label: int
    official_split: str
    protocol: str
    split: str
    group_id: str

