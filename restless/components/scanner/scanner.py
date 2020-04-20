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
colored = logging.colored
same_line = logging.same_line
flush = logging.flush

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
            logger.info("Prechecking " + colored("1", ["bold", "d_gray"]) + " file for metadata - {}.".format(path))
            result = await self.pea.analyze_file(path)
            flush(newline=True)
            results.append(result)
        else:
            count = 0
            flush(newline=True)
            for dirpath, dirs, files in os.walk(path):
                for filename in files:
                    fname = os.path.join(dirpath, filename)
                    # Unfortunately the logging lib doesn't support printing on
                    # the same line easily, so we have to flush the last printed line
                    logger.info(same_line("Prechecking {} files for metadata - {}.".format(colored(count, ["bold", "d_gray"]), colored(filename, ["underline", "d_gray"]))))
                    flush()
                    result = await self.pea.analyze_file(fname)
                    results.append(result)
                    count += 1
        return results
