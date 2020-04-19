import sys
import os

# make dep imports work when running in dir and in outside scripts
PACKAGE_PARENT = "../../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
try:
    from restless.components.utils import utils as utils
except Exception as e:
    from ..utils import utils as utils

from pe_analyzer import PEAnalyzer

pea = PEAnalyzer()

logging = utils.logger
logger = utils.logger.logger


class Scanner:
    """
    Extracts and analyzes information from files and classifies it for malware probability using NLP models.
    """

    def __init__(self):
        self.pea = pea
        return

    def scan_folder(self, folderpath: str) -> list:
        results = []
        # recursive walk
        for dirpath, dirs, files in os.walk(folderpath):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                result = self.pea.analyze_file(fname)
                if not result:
                    results.append(result)
        return results

    def scan_file(self, filepath: str) -> list:
        results = self.scan_folder(filepath)
        return results
