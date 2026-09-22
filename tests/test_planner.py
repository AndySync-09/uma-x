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
        self.assertEqual(plan.threads, 16)

    def test_reported_sycl_gpu_requests_control_default_offload(self):
        hw = HardwareSnapshot(
            system="Windows",
            release="11",
            machine="AMD64",
            processor="test",
            logical_cpus=22,
            total_ram_bytes=32 * 1024**3,
            llama_devices=("Found 1 SYCL devices:", "[level_zero:gpu:0] Intel Arc Graphics"),
        )
        plan = choose_plan(hw, 1024)
        self.assertEqual(plan.gpu_layers, 99)
        self.assertEqual(plan.threads, 16)
        self.assertEqual(plan.llama_args(), ["-t", "16", "-ngl", "99"])


if __name__ == "__main__":
    unittest.main()
