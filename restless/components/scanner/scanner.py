import sys
import os
import concurrent.futures
import asyncio

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

from multiprocessing import cpu_count


class Scanner:
    """
    Extracts and analyzes information from files and classifies it for malware probability using NLP models.
    """

    def __init__(self):
        self.pea = pea
        return

    def scan_file(self, path: str, flush_line: bool = False) -> list:
        logger.info(
            same_line(
                "Prechecking "
                + "file for metadata - {}.".format(colored(path, "d_gray"))
            )
        )
        if flush_line:
            flush()
        return self.pea.analyze_file(path)

    async def scan_recursive(self, path: str) -> list:
        results = []
        path = os.path.abspath(path)
        loop = asyncio.get_event_loop()
        if os.path.isfile(path):
            return self.scan_file(path)
        else:
            flush(newline=True)
            # recursive walk
            for dirpath, dirs, files in os.walk(path):
                # use multiprocessing
                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                    future_to_scan = [
                        loop.run_in_executor(
                            executor, self.scan_file, os.path.join(dirpath, file), True
                        )
                        for file in files
                    ]
                    if future_to_scan:
                        completed, pending = await asyncio.wait(future_to_scan)
                        results.extend([t.result() for t in completed])
        return results
