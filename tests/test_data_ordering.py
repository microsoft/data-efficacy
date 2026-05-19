import os
import sys
import unittest
from types import SimpleNamespace


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
ORDERING_ROOT = os.path.join(REPO_ROOT, "data_ordering")
sys.path.insert(0, ORDERING_ROOT)

import folding
import saw
import segment
import sorting
import stair
import zigzag


def sample_data(n_items=12):
    return [{"id": idx, "score": float(idx)} for idx in range(n_items)]


def ids(items):
    return [item["id"] for item in items]


class DataOrderingTest(unittest.TestCase):
    def test_sorting_orders_by_score(self):
        args = SimpleNamespace(score_field="score", ascending=True, use_gumbel=False, temperature=0, window_size=0)
        self.assertEqual(ids(sorting.order(sample_data(5)[::-1], args)), [0, 1, 2, 3, 4])

    def test_folding_matches_stride_partition(self):
        args = SimpleNamespace(score_field="score", ascending=True, folding_layer=3, window_size=0)
        self.assertEqual(ids(folding.order(sample_data(9), args)), [0, 3, 6, 1, 4, 7, 2, 5, 8])

    def test_zigzag_reverses_odd_folding_layers(self):
        args = SimpleNamespace(score_field="score", ascending=True, zigzag_layer=3, window_size=0)
        self.assertEqual(ids(zigzag.order(sample_data(9), args)), [0, 3, 6, 7, 4, 1, 2, 5, 8])

    def test_segment_preserves_dataset(self):
        args = SimpleNamespace(
            score_field="score",
            x_pct=25,
            y_pct=25,
            front_is_high=False,
            back_is_high=True,
            seed=1,
        )
        output = segment.order(sample_data(12), args)
        self.assertEqual(sorted(ids(output)), list(range(12)))
        self.assertTrue(set(ids(output[:3])).issubset({0, 1, 2}))
        self.assertTrue(set(ids(output[-3:])).issubset({9, 10, 11}))

    def test_stair_and_saw_preserve_dataset(self):
        args = SimpleNamespace(
            score_field="score",
            ascending=True,
            num_sections=3,
            folding_ratio=1 / 12,
            folding_layer=2,
            window_size=0,
        )
        data = sample_data(12)
        self.assertEqual(sorted(ids(stair.order(data, args))), list(range(12)))
        self.assertEqual(sorted(ids(saw.order(data, args))), list(range(12)))


if __name__ == "__main__":
    unittest.main()
