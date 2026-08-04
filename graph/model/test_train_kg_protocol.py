import numpy as np

from train_kg_protocol import calculate_metrics, stable_split


def test_group_split_is_stable():
    assert stable_split("A|B|C|D", 42) == stable_split("A|B|C|D", 42)


def test_metrics():
    y = np.array([0, 1, 2])
    logits = np.array([[3, 2, 1], [0, 2, 1], [2, 1, 3]], dtype=float)
    result = calculate_metrics(y, logits)
    assert result["top1"] == 1.0
    assert result["top3"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
