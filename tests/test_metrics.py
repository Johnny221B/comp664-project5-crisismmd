import pytest
from src.metrics import classification_metrics


def test_metrics_perfect():
    result = classification_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_metrics_all_negative_boundary():
    result = classification_metrics([0, 1], [0, 0])
    assert result["macro_f1"] == pytest.approx(1 / 3)

