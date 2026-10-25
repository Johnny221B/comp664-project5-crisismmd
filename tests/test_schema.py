import pytest
from src.data.schema import normalize_event, normalize_label


def test_binary_mapping():
    assert normalize_label("not_informative") == 0
    assert normalize_label("informative") == 1


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        normalize_label("dont_know_or_cant_judge")


def test_event_normalization():
    assert normalize_event("California Wildfires") == "california_wildfires"

