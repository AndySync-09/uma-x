import json
import unittest

from umax.benchmark import _extract_json, metric_summary, percent_delta


class BenchmarkTests(unittest.TestCase):
    def test_extract_and_summarize_llama_bench_json(self):
        payload = [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000.0},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 20.0},
        ]
        rows = _extract_json(json.dumps(payload))
        summary = metric_summary({"rows": rows})
        self.assertEqual(summary["prompt_tps"], 1000.0)
        self.assertEqual(summary["generation_tps"], 20.0)

    def test_percent_delta(self):
        self.assertEqual(percent_delta(10.0, 12.0), 20.0)
        self.assertIsNone(percent_delta(0.0, 12.0))


if __name__ == "__main__":
    unittest.main()
