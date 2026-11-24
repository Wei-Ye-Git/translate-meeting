import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import Config
from review_extension import load_review_extension


class ReviewExtensionTests(unittest.TestCase):
    def test_explicit_opt_out_does_not_import_an_extension_or_validate_its_config(self):
        with mock.patch.dict("os.environ", {
            "MEETING_COPILOT_ENABLED": "0", "MEETING_AI_TIMEOUT_SECONDS": "invalid",
        }, clear=True):
            config = Config()
        with mock.patch("review_extension.import_module") as importer:
            self.assertEqual(load_review_extension(config), (None, None))
        importer.assert_not_called()

    def test_explicit_opt_in_still_tolerates_unpublished_extension(self):
        with mock.patch.dict("os.environ", {"MEETING_COPILOT_ENABLED": "1"}, clear=True):
            config = Config()
        with mock.patch("review_extension.import_module", side_effect=ModuleNotFoundError("local_ai")):
            with mock.patch("builtins.print") as output:
                self.assertEqual(load_review_extension(config), (None, None))
        output.assert_called_once()

    def test_explicit_opt_in_selects_the_requested_provider(self):
        with mock.patch.dict("os.environ", {"MEETING_COPILOT_ENABLED": "1", "MEETING_AI_CLI": "opencode"}, clear=True):
            config = Config()
        with mock.patch("review_extension.import_module") as importer, mock.patch("builtins.print"):
            backend = importer.return_value
            cli = {"name": "OpenCode", "provider": "opencode"}
            backend.resolve_ai_cli.return_value = cli
            self.assertEqual(load_review_extension(config), (backend, cli))
        backend.resolve_ai_cli.assert_called_once_with("opencode")

    def test_core_session_starts_and_drains_without_the_optional_extension(self):
        import main

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict("os.environ", {"MEETING_COPILOT_ENABLED": "1"}, clear=True),
            mock.patch("review_extension.import_module", side_effect=ModuleNotFoundError("local_ai")),
            mock.patch.object(main, "pick_loopback_device", return_value=mock.Mock(name="device")),
            mock.patch.object(main, "missing_model_files", return_value=[]),
            mock.patch.object(main, "ASR"),
            mock.patch.object(main, "Translator"),
            mock.patch.object(main, "SileroSpeechDetector"),
            mock.patch.object(main, "create_session_directory", return_value=directory),
            mock.patch.object(main, "Overlay"),
            mock.patch.object(main, "capture_loop") as capture,
            mock.patch.object(main, "t", side_effect=lambda key, **_: key),
            mock.patch("builtins.print"),
        ):
            main.pick_loopback_device.return_value.name = "Test loopback"
            main.main()
            events = {
                row["event"]: row for row in map(
                    json.loads, Path(directory, "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
                )
            }
        capture.assert_called_once()
        self.assertFalse(events["session_started"]["realtime_copilot_enabled"])
        self.assertEqual(events["session_finished"]["workers_still_running"], [])
        self.assertEqual(events["session_finished"]["pending_history_translations"], 0)
