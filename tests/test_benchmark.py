import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from umax.benchmark import BenchmarkError, _extract_json, metric_summary, percent_delta, run_llama_bench


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

    def test_run_llama_bench_uses_files_not_subprocess_pipes(self):
        payload = [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000.0},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 20.0},
        ]

        def fake_run(command, **kwargs):
            self.assertIn("stdout", kwargs)
            self.assertIn("stderr", kwargs)
            self.assertNotIn("capture_output", kwargs)
            kwargs["stdout"].write(json.dumps(payload))
            kwargs["stderr"].write("sycl diagnostic")
            return SimpleNamespace(returncode=0)

        with patch("umax.benchmark.subprocess.run", side_effect=fake_run):
            result = run_llama_bench(
                "llama-bench",
                "model.gguf",
                prompt_tokens=512,
                gen_tokens=128,
                repetitions=2,
            )

        self.assertEqual(result["rows"], payload)
        self.assertEqual(result["stderr"], "sycl diagnostic")
        self.assertEqual(result["process_exit_code"], 0)
        self.assertIsNone(result["teardown_warning"])

    def test_windows_sycl_teardown_access_violation_is_tolerated_after_valid_json(self):
        payload = [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000.0},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 20.0},
        ]

        def fake_run(command, **kwargs):
            kwargs["stdout"].write(json.dumps(payload))
            kwargs["stderr"].write("sycl warning")
            return SimpleNamespace(returncode=3221225477)

        with patch("umax.benchmark.sys.platform", "win32"), \
             patch("umax.benchmark.subprocess.run", side_effect=fake_run):
            result = run_llama_bench(
                "llama-bench",
                "model.gguf",
                prompt_tokens=512,
                gen_tokens=128,
                repetitions=2,
            )

        self.assertEqual(result["rows"], payload)
        self.assertEqual(result["process_exit_code"], 3221225477)
        self.assertIn("0xC0000005", result["teardown_warning"])

    def test_windows_sycl_access_violation_still_fails_on_truncated_json(self):
        def fake_run(command, **kwargs):
            kwargs["stdout"].write('[{"n_prompt": 512')
            return SimpleNamespace(returncode=3221225477)

        with patch("umax.benchmark.sys.platform", "win32"), \
             patch("umax.benchmark.subprocess.run", side_effect=fake_run):
            with self.assertRaises(BenchmarkError):
                run_llama_bench(
                    "llama-bench",
                    "model.gguf",
                    prompt_tokens=512,
                    gen_tokens=128,
                    repetitions=2,
                )


if __name__ == "__main__":
    unittest.main()
