import unittest

from umax.memory import Access
from umax.prefetch import plan_prefetch


class PrefetchPlannerTests(unittest.TestCase):
    def test_prefetch_is_scheduled_before_tensor_use(self):
        accesses = [
            Access("A", 1024, 10),
            Access("B", 2048, 20),
        ]

        plan = plan_prefetch(accesses, lead_nodes=4)

        self.assertEqual(
            plan,
            [
                {
                    "tensor_id": "A",
                    "size_bytes": 1024,
                    "prefetch_seq": 6,
                    "use_seq": 10,
                    "lead_nodes": 4,
                },
                {
                    "tensor_id": "B",
                    "size_bytes": 2048,
                    "prefetch_seq": 16,
                    "use_seq": 20,
                    "lead_nodes": 4,
                },
            ],
        )

    def test_prefetch_never_schedules_before_graph_start(self):
        accesses = [
            Access("A", 1024, 2),
        ]

        plan = plan_prefetch(accesses, lead_nodes=8)

        self.assertEqual(plan[0]["prefetch_seq"], 0)
        self.assertEqual(plan[0]["lead_nodes"], 2)

    def test_repeated_tensor_uses_create_distinct_predictions(self):
        accesses = [
            Access("A", 1024, 10),
            Access("A", 1024, 30),
        ]

        plan = plan_prefetch(accesses, lead_nodes=5)

        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["use_seq"], 10)
        self.assertEqual(plan[1]["use_seq"], 30)


if __name__ == "__main__":
    unittest.main()
