
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import unittest

from restless import restless

class TestRestlessSuite(unittest.TestCase):
    
    def test_restless_import(self):
       from restless import restless
       self.assertIsNotNone(restless)

if __name__ == '__main__':
    unittest.main()
