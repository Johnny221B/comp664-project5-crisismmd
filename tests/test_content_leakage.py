import pandas as pd
import pytest

from scripts.validate_content_leakage import validate_content


def test_cross_split_checksum_rejected():
    frame = pd.DataFrame({
        "split": ["train", "test"],
        "image_checksum": ["same", "same"],
    })
    with pytest.raises(ValueError, match="content leakage"):
        validate_content(frame, "fixture")


def test_distinct_checksum_passes():
    frame = pd.DataFrame({
        "split": ["train", "val", "test"],
        "image_checksum": ["a", "b", "c"],
    })
    validate_content(frame, "fixture")
