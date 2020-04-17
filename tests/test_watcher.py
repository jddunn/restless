import unittest
import os

from restless import restless

r = restless()
watcher = r.watcher()
print(watcher)

class WatcherTest(unittest.TestCase):

    def setUp(self):
        r = restles()
        self.watcher = r.watcher()
        return

    def test_watcher_init(self):
        assert self.watcher is not None

if __name__ == "__main__":
    unittest.main()
