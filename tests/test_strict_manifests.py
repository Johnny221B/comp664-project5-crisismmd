import pandas as pd

from scripts.make_strict_manifests import make_strict


def test_evaluation_split_priority_is_explicit():
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"],
        "split": ["train", "val", "test", "train"],
        "image_checksum": ["same", "same", "same", "unique"],
        "protocol": ["raw"] * 4,
    })
    # Avoid filesystem hashing because this fixture supplies checksums directly.
    priority = {"train": 0, "val": 1, "test": 2}
    keep = frame.groupby("image_checksum").split.transform(lambda x: max(x, key=priority.__getitem__))
    result = frame[frame.split.eq(keep)]
    assert set(result.sample_id) == {"c", "d"}
