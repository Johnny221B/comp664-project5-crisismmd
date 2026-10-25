import pandas as pd
import pytest
from scripts.validate_splits import validate_frame


def frame(rows, protocol="ood_heldout"):
    result = pd.DataFrame(rows, columns=["sample_id", "tweet_id", "image_id", "event", "label", "split"])
    result["protocol"] = protocol
    return result


def test_valid_ood_split():
    validate_frame(frame([
        ("a", "t1", "i1", "hurricane_harvey", 0, "train"),
        ("b", "t2", "i2", "hurricane_irma", 1, "val"),
        ("c", "t3", "i3", "california_wildfires", 1, "test"),
    ]))


def test_california_train_leakage_rejected():
    data = frame([
        ("a", "t1", "i1", "california_wildfires", 0, "train"),
        ("b", "t2", "i2", "california_wildfires", 1, "test"),
    ])
    with pytest.raises(ValueError, match="California"):
        validate_frame(data)


def test_california_train_leakage_rejected_for_strict_protocol():
    data = frame([
        ("a", "t1", "i1", "california_wildfires", 0, "train"),
        ("b", "t2", "i2", "california_wildfires", 1, "test"),
    ], protocol="ood_heldout_strict")
    with pytest.raises(ValueError, match="California"):
        validate_frame(data)


def test_disagreed_labels_rejected():
    data = frame([
        ("a", "t1", "i1", "hurricane_harvey", 0, "train"),
        ("b", "t2", "i2", "hurricane_irma", 1, "val"),
        ("c", "t3", "i3", "california_wildfires", 1, "test"),
    ])
    data["text_label"] = [0, 1, 1]
    data["image_label"] = [1, 1, 1]
    with pytest.raises(ValueError, match="label disagreement"):
        validate_frame(data)


def test_group_overlap_rejected():
    data = frame([
        ("a", "t1", "i1", "hurricane_harvey", 0, "train"),
        ("b", "t1", "i2", "hurricane_irma", 1, "val"),
        ("c", "t3", "i3", "california_wildfires", 1, "test"),
    ])
    with pytest.raises(ValueError, match="tweet_id overlap"):
        validate_frame(data)
