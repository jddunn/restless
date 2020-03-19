# Make relative imports work for Docker
import sys
import os

PACKAGE_PARENT = "."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from components.utils import Utils
from components.watcher import Watcher
from components.watcher import Scanner

utils = Utils()
watcher = Watcher()
scanner = Scanner()

class Restless:
    """
    Main Restless module.
    """

    def __init__(self):
        print("I'm restless")
