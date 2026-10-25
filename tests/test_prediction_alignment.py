import pandas as pd
import pytest


def assert_aligned(left, right):
    if not left.sort_values("sample_id").sample_id.reset_index(drop=True).equals(right.sort_values("sample_id").sample_id.reset_index(drop=True)):
        raise ValueError("sample_id alignment mismatch")


def test_alignment_rejects_missing_id():
    with pytest.raises(ValueError, match="alignment"):
        assert_aligned(pd.DataFrame({"sample_id": ["a", "b"]}), pd.DataFrame({"sample_id": ["a"]}))


def test_alignment_accepts_reordering():
    assert_aligned(pd.DataFrame({"sample_id": ["a", "b"]}), pd.DataFrame({"sample_id": ["b", "a"]}))
