import tempfile
from pathlib import Path
import unittest

from umax.hardware import HardwareSnapshot
from umax.profile import ProfileStore, model_fingerprint


class ProfileTests(unittest.TestCase):
    def test_profile_is_machine_and_model_scoped(self):
        hw = HardwareSnapshot(
            system="Windows",
            release="11",
            machine="AMD64",
            processor="test",
            logical_cpus=8,
            total_ram_bytes=16 * 1024**3,
            llama_devices=(),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "tiny.gguf"
            model.write_bytes(b"GGUF-test-data")
            store = ProfileStore(root / "profiles")
            result = store.record(model, hw, {"type": "test"})
            self.assertTrue(result.is_file())
            self.assertTrue(model_fingerprint(model) in result.name)


if __name__ == "__main__":
    unittest.main()
