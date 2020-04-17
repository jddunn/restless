import unittest
import os

from restless.components.utils import Utils

class LoggerTest(unittest.TestCase):

    def setUp(self):
        utils = Utils()
        self.logger = utils.logger

    def test_logger_init(self):
        self.assertIsNotNone(self.logger)

    def test_colored_text(self):
        text = "color this!"
        colored_text = self.logger.colored(text, "red")
        self.assertEqual(colored_text, "\x1b[31mcolor this!\x1b[0m")

if __name__ == "__main__":
    unittest.main()

