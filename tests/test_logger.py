
import unittest
import os

from restless.components.utils import Utils

class LoggerTest(unittest.TestCase):

    def setUp(self):
        utils = Utils()
        self.logger = utils.logger
        return

    def test_logger_init(self):
        self.assertIsNotNone(self.logger)
        return

if __name__ == "__main__":
    unittest.main()

