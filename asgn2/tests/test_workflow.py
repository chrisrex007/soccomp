"""Regression checks for checkpoint recovery and report calculations.

Run with: python -m unittest discover -s asgn2/tests -v
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Assignment2_23CS10012_analysis_task4 as workflow
import Assignment2_23CS10012_paper_task2 as paper
from Assignment2_23CS10012_greedy_task1 import graph_matrix, save_json


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids, cls.degree, cls.P = graph_matrix(nx.path_graph(6))
        cls.alpha = np.full(6, 0.5)
        cls.opinions = np.array([0.1, 0.7, 0.2, 0.9, 0.4, 0.6])
        cls.random = paper.random_trials(
            cls.P, cls.alpha, cls.opinions, cls.ids, k=3, trials=20
        )

    def test_random_completion_is_written_only_with_aggregates(self):
        snapshots = []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "random.json"

            def capture(destination, value):
                save_json(destination, value)
                snapshots.append(json.loads(path.read_text()))

            with patch.object(paper, "save_json", side_effect=capture):
                paper.random_trials(
                    self.P, self.alpha, self.opinions, self.ids, k=3, out=path
                )

        self.assertTrue(snapshots[-1]["complete"])
        self.assertTrue(all(not row["complete"] for row in snapshots[:-1]))
        workflow.validate_budgeted_result(snapshots[-1], 3, trials=20)

    def test_interruption_after_last_trial_leaves_an_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "random.json"

            def interrupt(destination, value):
                save_json(destination, value)
                if len(value["trials"]) == 20 and "rows" not in value:
                    raise InterruptedError("simulated interruption before aggregation")

            with patch.object(paper, "save_json", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    paper.random_trials(
                        self.P, self.alpha, self.opinions, self.ids, k=3, out=path
                    )
            checkpoint = json.loads(path.read_text())
            self.assertFalse(checkpoint["complete"])
            with self.assertRaises(ValueError):
                workflow.validate_budgeted_result(checkpoint, 3, trials=20)

    def test_short_complete_runs_are_rejected(self):
        result = paper.fast(
            self.P, self.alpha, self.opinions, self.ids, np.arange(6), 3
        )
        self.assertTrue(result["complete"])
        with self.assertRaises(ValueError):
            workflow.validate_budgeted_result(result, 30)

    def test_random_validation_requires_every_trial_and_its_aggregate(self):
        workflow.validate_budgeted_result(self.random, 3, trials=20)
        missing_trial = deepcopy(self.random)
        missing_trial["trials"].pop()
        short_trial = deepcopy(self.random)
        short_trial["trials"][-1].pop()
        wrong_mean = deepcopy(self.random)
        wrong_mean["rows"][0]["f"] += 1
        for result in (missing_trial, short_trial, wrong_mean):
            with self.subTest(result=result):
                with self.assertRaises(ValueError):
                    workflow.validate_budgeted_result(result, 3, trials=20)

    def test_local_validation_requires_all_starts_and_finite_diagnostics(self):
        result = paper.unbudgeted(self.P, self.alpha, self.opinions, self.ids, -1)
        workflow.validate_local_result(result, 6)
        missing_start = deepcopy(result)
        missing_start["runs"].pop()
        invalid_bound = deepcopy(result)
        invalid_bound["runs"][0]["objective_error_bound"] = float("nan")
        wrong_reference = deepcopy(result)
        wrong_reference["optimum"]["f"] += 1
        for invalid in (missing_start, invalid_bound, wrong_reference):
            with self.subTest(result=invalid):
                with self.assertRaises(ValueError):
                    workflow.validate_local_result(invalid, 6)

    def test_report_uses_saved_arrays_and_checks_original_id_order(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            np.savez(
                output / "inputs.npz",
                ids=self.ids,
                alpha=self.alpha,
                U=self.opinions,
                P=1 - self.opinions,
            )
            graph = (self.ids, self.degree, self.P, {})
            with patch.object(workflow, "load_graph", return_value=graph):
                _, alpha, profiles = workflow.load_report_inputs("unused", output)
                np.testing.assert_array_equal(alpha, self.alpha)
                np.testing.assert_array_equal(profiles["U"], self.opinions)
                np.testing.assert_array_equal(profiles["P"], 1 - self.opinions)
                np.savez(
                    output / "inputs.npz",
                    ids=self.ids[::-1],
                    alpha=self.alpha,
                    U=self.opinions,
                    P=1 - self.opinions,
                )
                with self.assertRaises(ValueError):
                    workflow.load_report_inputs("unused", output)

    def test_completed_results_reject_incomplete_or_different_source_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            markers = (
                {"complete": False, "experiment_hash": "current"},
                {"complete": True, "experiment_hash": "old"},
            )
            with patch.object(workflow, "experiment_hash", return_value="current"):
                for marker in markers:
                    save_json(output / "complete.json", marker)
                    with self.subTest(marker=marker):
                        with self.assertRaises(RuntimeError):
                            workflow.load_completed_results("unused", output)

    def test_report_ratios_have_the_same_meaning_for_max_and_min(self):
        analysis = workflow.calculate_comparisons(comparison_fixture())
        self.assertEqual(len(analysis["table"]), 36)
        self.assertEqual(len(analysis["comparisons"]), 32)
        for row in analysis["table"]:
            if row["method"] == "fast" and row["k"] == 5:
                self.assertAlmostEqual(row["gain_fraction"], 0.1)
        for row in analysis["comparisons"]:
            if row["method"] == "fast":
                self.assertAlmostEqual(row["gain_ratio"], 0.5)
                self.assertEqual(row["jaccard"], 1)


def comparison_fixture():
    """Known arithmetic examples for report math, not experiment results."""
    results = {}
    for profile, direction in workflow.SCENARIOS:
        sign = 1 if direction == "max" else -1
        results[f"local_{profile}_{direction}_all.json"] = {
            "optimum": {"f": 10 + sign * 10}
        }
        for method, rate in (
            ("fast", 0.2), ("I", 0.1), ("II", 0.05), ("random", 0.075)
        ):
            rows = [
                dict(
                    k=k,
                    added=k - 1,
                    f=10 + sign * rate * k,
                    gain=rate,
                    seconds=k / 10,
                    peak_mib=1,
                )
                for k in range(1, 31)
            ]
            result = {"initial": 10, "rows": rows}
            if method == "random":
                result["trials"] = [deepcopy(rows) for _ in range(20)]
            results[f"{method}_{profile}_{direction}_all.json"] = result
            if direction == "max":
                results[f"{method}_{profile}_max_C.json"] = result
        if direction == "max":
            results[f"groundtruth_{profile}.json"] = {
                "initial": 10,
                "references": {
                    str(k): {"f": 10 + 0.4 * k, "ids": list(range(k))}
                    for k in (2, 3)
                },
            }
    return results


if __name__ == "__main__":
    unittest.main()
