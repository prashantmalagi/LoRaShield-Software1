import unittest

from morse_correction_v2.data import make_example
from morse_correction_v2.metrics import evaluate_gate


class V2MetricsTests(unittest.TestCase):
    def test_metric_report_includes_denominators_abstentions_and_boundaries(self):
        clean = make_example("cal-0", "calibration", "SOS HI", "clean", 1)
        corrupted = make_example("cal-1", "calibration", "HELLO WORLD", "symbol_insertion", 2)
        rows = [clean, corrupted]
        report = evaluate_gate(rows, [clean.received_morse, corrupted.received_morse], [False, False])
        self.assertEqual(report["denominators"]["all_cases"], 2)
        self.assertEqual(report["denominators"]["clean_cases"], 1)
        self.assertEqual(report["denominators"]["abstentions"], 2)
        self.assertEqual(report["clean_false_accept_rate"], 0)
        for name in ("symbol_denominator", "letter_boundary_denominator", "word_boundary_denominator"):
            self.assertGreater(report["gated"][name], 0)


if __name__ == "__main__":
    unittest.main()
