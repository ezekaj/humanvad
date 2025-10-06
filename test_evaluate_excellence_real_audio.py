import unittest
from pathlib import Path

from evaluate_excellence_real_audio import DEFAULT_METADATA, evaluate


class EvaluateExcellenceRealAudioTest(unittest.TestCase):
    def test_default_fixtures_summary(self) -> None:
        summary = evaluate(Path("test_audio"), DEFAULT_METADATA)

        self.assertEqual(summary["samples_evaluated"], 10)
        self.assertEqual(summary["samples_skipped"], 0)

        excellence = summary["excellence"]
        production = summary["production"]

        self.assertAlmostEqual(excellence["accuracy"], 1.0, places=5)
        self.assertAlmostEqual(excellence["precision"], 1.0, places=5)
        self.assertAlmostEqual(excellence["recall"], 1.0, places=5)
        self.assertEqual(excellence["counts"], {"tp": 7, "fp": 0, "fn": 0, "tn": 3})

        self.assertAlmostEqual(production["accuracy"], 0.7, places=5)
        self.assertAlmostEqual(production["precision"], 0.7, places=5)
        self.assertAlmostEqual(production["recall"], 1.0, places=5)
        self.assertEqual(production["counts"], {"tp": 7, "fp": 3, "fn": 0, "tn": 0})

        self.assertGreater(summary["avg_excellence_turn_end_prob"], 0.7)
        self.assertGreater(summary["avg_decision_margin"], 0.18)


if __name__ == "__main__":
    unittest.main()
