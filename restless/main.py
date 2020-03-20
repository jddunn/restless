# Make relative imports work for Docker
import sys
import os

PACKAGE_PARENT = "."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from components.utils import utils
from components.watcher import Watcher
from components.scanner import Scanner
from components.nlp import nlp

class Restless(object):
    """
    Main Restless module.
    """

    def __init__(self, run_system_scan=False):
        self.run_system_scan = run_system_scan
        utils.print_logm(
            "Restless initializing. Running system-wide scan: "
            + str(self.run_system_scan)
        )
        self.scanner = Scanner()
        if self.run_system_scan:
            # Get last system scan time
            self.scanner.scan_full_system()
        else:
            self.watcher = Watcher()
        return
