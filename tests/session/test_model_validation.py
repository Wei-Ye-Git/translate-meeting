import os
"""Local model file validation tests."""

import tempfile
import unittest

from model_validation import MODEL_FILES, missing_model_files


class ModelValidationTests(unittest.TestCase):
    def test_reports_missing_required_model_files(self):
        with tempfile.TemporaryDirectory() as models_dir:
            missing = missing_model_files(models_dir)

        expected = sum(len(files) for files in MODEL_FILES.values())
        self.assertEqual(len(missing), expected)
        self.assertIn(os.path.join("whisper_base_en", "model.bin"), missing)

    def test_accepts_complete_model_directories(self):
        with tempfile.TemporaryDirectory() as models_dir:
            for model_dir, files in MODEL_FILES.items():
                path = os.path.join(models_dir, model_dir)
                os.makedirs(path)
                for filename in files:
                    with open(os.path.join(path, filename), "wb") as model_file:
                        model_file.write(b"model")

            self.assertEqual(missing_model_files(models_dir), [])

    def test_reports_empty_required_model_files(self):
        with tempfile.TemporaryDirectory() as models_dir:
            for model_dir, files in MODEL_FILES.items():
                path = os.path.join(models_dir, model_dir)
                os.makedirs(path)
                for filename in files:
                    with open(os.path.join(path, filename), "wb") as model_file:
                        model_file.write(b"" if filename == files[0] else b"model")

            self.assertIn(
                os.path.join(next(iter(MODEL_FILES)), next(iter(MODEL_FILES.values()))[0]),
                missing_model_files(models_dir),
            )


if __name__ == "__main__":
    unittest.main()
