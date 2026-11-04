import os
import subprocess
import sys
import unittest


class StartupEncodingTests(unittest.TestCase):
    def test_python_can_write_chinese_startup_log(self):
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "-c", "print('加载 Whisper')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "加载 Whisper")


if __name__ == "__main__":
    unittest.main()
