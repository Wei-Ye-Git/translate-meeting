import queue
import unittest

from translation_queue import publish_live_translation, take_latest


class TranslationQueueTests(unittest.TestCase):
    def test_prefers_newest_pending_translation(self):
        pending = queue.Queue()
        pending.put((1, "old stable text"))
        pending.put((2, "new stable text"))

        latest = take_latest((0, "current text"), pending)

        self.assertEqual(latest, (2, "new stable text"))
        self.assertTrue(pending.empty())

    def test_returns_current_item_when_nothing_newer_is_pending(self):
        pending = queue.Queue()
        item = (3, "stable text")

        self.assertIs(take_latest(item, pending), item)

    def test_does_not_retranslate_unchanged_stable_text(self):
        pending = queue.Queue(maxsize=1)
        state = {"revision": 0}

        first = publish_live_translation("The stable sentence", 2, state, pending)
        repeated = publish_live_translation("The stable sentence", 2, state, pending)

        self.assertEqual(first, 1)
        self.assertIsNone(repeated)
        self.assertEqual(pending.get_nowait(), (1, "The stable sentence"))

    def test_new_stable_text_replaces_an_obsolete_pending_translation(self):
        pending = queue.Queue(maxsize=1)
        state = {"revision": 0}

        publish_live_translation("The first sentence", 2, state, pending)
        revision = publish_live_translation("The second sentence", 2, state, pending)

        self.assertEqual(revision, 2)
        self.assertEqual(pending.get_nowait(), (2, "The second sentence"))


if __name__ == "__main__":
    unittest.main()
