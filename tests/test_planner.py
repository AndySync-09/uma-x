import unittest

from umax.hardware import HardwareSnapshot
from umax.planner import choose_plan


class PlannerTests(unittest.TestCase):
    def test_cpu_only_stays_on_cpu(self):
        hw = HardwareSnapshot(
            system="Windows",
            release="11",
            machine="AMD64",
            processor="test",
            logical_cpus=16,
            total_ram_bytes=32 * 1024**3,
            llama_devices=("CPU: test",),
        )
        plan = choose_plan(hw, 1024)
        self.assertEqual(plan.gpu_layers, 0)
        self.assertEqual(plan.threads, 8)

    def test_reported_gpu_requests_offload(self):
        hw = HardwareSnapshot(
            system="Windows",
            release="11",
            machine="AMD64",
            processor="test",
            logical_cpus=20,
            total_ram_bytes=32 * 1024**3,
            llama_devices=("SYCL0: Intel Arc GPU",),
        )
        plan = choose_plan(hw, 1024)
        self.assertEqual(plan.gpu_layers, -1)
        self.assertEqual(plan.flash_attention, "auto")


if __name__ == "__main__":
    unittest.main()
