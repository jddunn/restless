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

    async def scan_recursive(self, path: str) -> list:
        results = []
        path = os.path.abspath(path)
        # recursive walk
        if os.path.isfile(path):
            result = self.pea.analyze_file(path)
            results.append(result)
        else:
            for dirpath, dirs, files in os.walk(path):
                for filename in files:
                    fname = os.path.join(dirpath, filename)
                    result = self.pea.analyze_file(fname)
                    results.append(result)
        return results
