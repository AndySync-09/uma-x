import json
from pathlib import Path
import tempfile
import unittest

from umax.trace import analyze_trace, load_trace


class TraceAnalyzerTests(unittest.TestCase):
    def test_load_trace_reads_valid_jsonl(self):
        rows = [
            {
                "v": 2,
                "event": "node",
                "seq": 0,
                "split": 0,
                "node": 0,
                "backend": "CPU",
                "tensor_id": "a",
                "name": "inp_embd",
                "op": "GET_ROWS",
                "type": "f32",
                "bytes": 8192,
                "ne": [2048, 1, 1, 1],
                "buffer": "SYCL_Host",
                "buffer_type": "SYCL_Host",
                "buffer_size": 100000,
                "buffer_offset": 0,
                "data_id": "x",
                "view_src_id": None,
                "view_offs": 0,
                "sources": [],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            self.assertEqual(load_trace(path), rows)

    def test_analyze_trace_reports_core_memory_signals(self):
        rows = [
            {
                "v": 2,
                "event": "node",
                "seq": 0,
                "split": 0,
                "node": 0,
                "backend": "SYCL0",
                "tensor_id": "a",
                "name": "x",
                "op": "MUL",
                "type": "f32",
                "bytes": 1024,
                "ne": [256, 1, 1, 1],
                "buffer": "SYCL0",
                "buffer_type": "SYCL0",
                "buffer_size": 4096,
                "buffer_offset": 0,
                "data_id": "p0",
                "view_src_id": None,
                "view_offs": 0,
                "sources": [
                    {
                        "id": "w",
                        "name": "blk.0.weight",
                        "type": "f16",
                        "bytes": 2048,
                        "buffer": "SYCL0",
                        "buffer_type": "SYCL0",
                        "buffer_size": 4096,
                        "buffer_offset": 2048,
                        "data_id": "pw",
                        "view_src_id": None,
                        "view_offs": 0,
                    }
                ],
            },
            {
                "v": 2,
                "event": "node",
                "seq": 1,
                "split": 0,
                "node": 1,
                "backend": "SYCL0",
                "tensor_id": "b",
                "name": "y",
                "op": "ADD",
                "type": "f32",
                "bytes": 1024,
                "ne": [256, 1, 1, 1],
                "buffer": "SYCL0",
                "buffer_type": "SYCL0",
                "buffer_size": 4096,
                "buffer_offset": 0,
                "data_id": "p0",
                "view_src_id": None,
                "view_offs": 0,
                "sources": [
                    {
                        "id": "w",
                        "name": "blk.0.weight",
                        "type": "f16",
                        "bytes": 2048,
                        "buffer": "SYCL0",
                        "buffer_type": "SYCL0",
                        "buffer_size": 4096,
                        "buffer_offset": 2048,
                        "data_id": "pw",
                        "view_src_id": None,
                        "view_offs": 0,
                    }
                ],
            },
        ]

        report = analyze_trace(rows)

        self.assertEqual(report["records"], 2)
        self.assertEqual(report["backends"], {"SYCL0": 2})
        self.assertEqual(report["unique_weights"], 1)
        self.assertEqual(report["physical_slots"], 1)
        self.assertEqual(report["reused_slots"], 1)
        self.assertEqual(report["next_use"]["observations"], 1)
        self.assertEqual(report["next_use"]["min"], 1)
        self.assertEqual(report["next_use"]["max"], 1)


if __name__ == "__main__":
    unittest.main()
