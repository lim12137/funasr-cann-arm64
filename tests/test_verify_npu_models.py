import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "verify" / "verify_npu_models.py"
SPEC = importlib.util.spec_from_file_location("verify_npu_models", MODULE_PATH)
verify_npu_models = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_npu_models
SPEC.loader.exec_module(verify_npu_models)


class RequiredDirectoryTests(unittest.TestCase):
    def test_returns_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = verify_npu_models.required_directory("NANO_MODEL_DIR", directory)

        self.assertEqual(result, Path(directory))

    def test_rejects_missing_path(self):
        with self.assertRaisesRegex(RuntimeError, "NANO_MODEL_DIR"):
            verify_npu_models.required_directory("NANO_MODEL_DIR", "/missing/nano")

    def test_rejects_file_path(self):
        with tempfile.NamedTemporaryFile() as file:
            with self.assertRaisesRegex(RuntimeError, "NANO_MODEL_DIR"):
                verify_npu_models.required_directory("NANO_MODEL_DIR", file.name)


class TranscriptionTextTests(unittest.TestCase):
    def test_returns_first_non_empty_text(self):
        text = verify_npu_models.transcription_text([{"text": "  validation passed  "}])

        self.assertEqual(text, "validation passed")

    def test_rejects_empty_result(self):
        with self.assertRaisesRegex(RuntimeError, "empty transcription"):
            verify_npu_models.transcription_text([])

    def test_rejects_blank_text(self):
        with self.assertRaisesRegex(RuntimeError, "empty transcription"):
            verify_npu_models.transcription_text([{"text": "   "}])


if __name__ == "__main__":
    unittest.main()
