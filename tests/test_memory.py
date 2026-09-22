import unittest

from umax.memory import Access, simulate_residency


class MemoryPolicyTests(unittest.TestCase):
    def test_next_use_policy_evicts_tensor_needed_farthest_in_future(self):
        accesses = [
            Access("A", 100, 0),
            Access("B", 100, 1),
            Access("C", 100, 2),
            Access("A", 100, 3),
            Access("C", 100, 4),
        ]

        report = simulate_residency(accesses, capacity_bytes=200)

        self.assertEqual(report["capacity_bytes"], 200)
        self.assertLessEqual(report["peak_resident_bytes"], 200)

        # When C arrives at seq 2, A is needed again before B.
        # B should therefore be evicted.
        self.assertIn(
            {"seq": 2, "tensor_id": "B"},
            report["evictions"],
        )

    def test_cache_hits_are_counted(self):
        accesses = [
            Access("A", 100, 0),
            Access("B", 100, 1),
            Access("A", 100, 2),
        ]

        report = simulate_residency(accesses, capacity_bytes=200)

        self.assertEqual(report["hits"], 1)
        self.assertEqual(report["misses"], 2)
        self.assertEqual(report["evictions"], [])


if __name__ == "__main__":
    unittest.main()
