#!/usr/bin/env python3
from __future__ import annotations

import unittest
from collections import deque

from scripts.data.prepare_datasets import normalize_role, pop_ready_candidates_in_source_order


class PrepareDatasetsRegressionTest(unittest.TestCase):
    def test_normalize_role_maps_tool_output_to_tool(self) -> None:
        self.assertEqual(normalize_role("tool_output"), "tool")

    def test_pop_ready_candidates_in_source_order(self) -> None:
        candidate_order = deque([3, 5, 7])
        completed_results = {
            5: {"source_index": 5},
            7: None,
        }

        self.assertEqual(pop_ready_candidates_in_source_order(candidate_order, completed_results), [])

        completed_results[3] = {"source_index": 3}
        ready = pop_ready_candidates_in_source_order(candidate_order, completed_results)

        self.assertEqual(
            ready,
            [
                (3, {"source_index": 3}),
                (5, {"source_index": 5}),
                (7, None),
            ],
        )
        self.assertEqual(list(candidate_order), [])
        self.assertEqual(completed_results, {})


if __name__ == "__main__":
    unittest.main()
