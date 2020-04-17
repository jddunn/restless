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
from components.nlp import NLP

logger = utils.logger

class Restless(object):
    """
    Main Restless module.
    """

    def __init__(self, run_system_scan=False):
        self.run_system_scan = run_system_scan
        # keys of features from pe_analyzer
        logger.print_log({"level": "info", "text":
          "Restless initializing. Running system-wide scan: "
           + str(self.run_system_scan)}
        )
        # logger.logger.success("HELP MEEEEE")
        self.scanner = Scanner()
        if self.run_system_scan:
            # Get last system scan time
            self.scanner.scan_full_system()
        else:
            self.watcher = Watcher([])
        nlp = NLP(load_default_hann_model=True)
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
            if len(self.nlp.hann.features) > 0:
                features = [x for x in features if x in self.nlp.hann.features]
            matrix_results = self.nlp.hann.build_feature_matrix_from_input_arr(features)
            res = (fname, self.nlp.hann.predict(matrix_results))
            results.append(res)
            # a_map = self.nlp.hann.attention_map(matrix_results)
            logger.print_logm(
                "Scanned {} - predicted: {}% benign and {}% malicious".format(
                    res[0], res[1][0], res[1][1]
                )
            )
        return results
