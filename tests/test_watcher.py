import unittest
import os

from restless.components.watcher import Watcher

class WatcherTest(unittest.TestCase):

    def setUp(self):
        self.watcher = Watcher(watch_pool=[])
        return

    def test_watcher_init(self):
        self.assertIsNotNone(self.watcher)
        return

if __name__ == "__main__":
    unittest.main()
