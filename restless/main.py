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
        # keys of features from pe_analyzer
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
        self.nlp = nlp
        return

    def scan_full_system(self):
        results = self.scanner.scan_full_system()
        return results

    def scan_folder(self, filepath: str):
        results = []
        file_results = self.scanner.scan_folder(filepath)
        for file_result in file_results:
            fname = file_result[0]
            features = file_result[1]
            matrix_results = self.nlp.hann.build_features_vecs_from_input(features)
            # print("Predicting: ", fname)
            res = (fname, self.nlp.hann.predict(matrix_results))
            results.append(res)
            utils.print_logm(
                "Scanned "
                + str(res[0])
                + " - predicted: "
                + str(res[1][0])
                + " benign and "
                + str(res[1][1])
                + " malicious"
            )
        return results
