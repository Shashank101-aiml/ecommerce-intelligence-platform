import pandas as pd

from src.features.rfm_features import _segment


def test_segment_thresholds():
    assert _segment(12) == "Champions"
    assert _segment(9) == "Champions"
    assert _segment(8) == "Loyal"
    assert _segment(7) == "Loyal"
    assert _segment(6) == "At Risk"
    assert _segment(5) == "At Risk"
    assert _segment(4) == "Lost"
    assert _segment(3) == "Lost"
