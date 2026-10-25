import pandas as pd

from scripts.statistical_analysis import holm_adjust, load_aligned


def test_holm_adjustment_is_monotone_in_rank():
    adjusted = holm_adjust(pd.Series([0.01, 0.04, 0.20]))
    assert adjusted.tolist() == [0.03, 0.08, 0.20]


def test_seed42_outcomes_are_exhaustive():
    expected = {"id_official_strict": 1534, "ood_heldout_strict": 1195}
    allowed = {"all_correct", "all_wrong", "fusion_rescue", "fusion_regression"}
    for protocol, total in expected.items():
        frame = load_aligned(protocol, 42)
        assert len(frame) == total
        assert set(frame.outcome) <= allowed
        assert frame.outcome.value_counts().sum() == total
