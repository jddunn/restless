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

misc = utils.misc
logging = utils.logger
logger = utils.logger.logger


class Restless(object):
    """
    Main Restless module.
    """

    def __init__(self, run_system_scan=True, constant_watch=False, watch_pool=["*"]):

        self.run_system_scan = run_system_scan
        logger.info("Restless initializing..")
        self.scanner = Scanner()
        self.watcher = Watcher(watch_pool)
        self.nlp = NLP(load_default_hann_model=True)
        if self.run_system_scan:
            logger.info("Scanning full system.")
            self.scan_full_system()
        if constant_watch:
            logger.info("Constantly watching: {}.".format(watch_pool))
            self.constant_watch(watch_pool)
        return

    def clean_files(self, infected_files: list) -> None:
        for file in infected_files:
            pass
        return

    def scan_full_system(self):
        root = misc.get_os_root_path()
        results = self.scanner.scan(root)
        return results

    def scan(self, filepath: str):
        results = []
        potential_malware = []
        file_results = self.scanner.scan_folder(filepath)
        for file_result in file_results:
            fname = file_result[0]
            short_fname = fname.split("/")[len(fname.split("/")) - 1]
            features = file_result[1]
            if len(self.nlp.hann.features) > 0:
                features = [x for x in features if x in self.nlp.hann.features]
            matrix_results = self.nlp.hann.build_feature_matrix_from_input_arr(features)
            result = (fname, self.nlp.hann.predict(matrix_results))
            results.append(result)
            colored_fname = logging.colored(short_fname, "gray")
            benign = float(result[1][0])
            malicious = float(result[1][1])
            # Colorize percentages
            colored_benign = (
                logging.colored(benign, "gray")
                + logging.colored("%", "gray")
                + " "
                + logging.colored("benign", "gray")
            )
            colored_malicious = (
                logging.colored(malicious, "gray")
                + logging.colored("%", "gray")
                + " "
                + logging.colored("malicious", "gray")
            )
            if benign > 0.6:
                clr = "green" if benign < 0.8 else "b_green"
                colored_benign = logging.colored(benign, clr)
                colored_benign += logging.colored("% benign", clr)
            if malicious > 0.1 and malicious < 0.4:
                clr = "yellow"
                colored_malicious = logging.colored(malicious, clr)
                colored_malicious += logging.colored("% malicious", clr)
            if malicious > 0.6:
                potential_malware.append(fname)
                clr = "red" if malicious > 0.8 else "b_red"
                colored_malicious = logging.colored(malicious, clr)
                colored_malicious += logging.colored("% malicious", clr)
            logger.info(
                "{} {} {} predicted: {} and {}.".format(
                    logging.colored("Scanned", "white"),
                    colored_fname,
                    logging.colored("-", "d_gray"),
                    colored_benign,
                    colored_malicious,
                )
            )
        if len(potential_malware) > 0:
            logger.critical(
                "Found {} files to be potentially infected!".format(
                    logging.colored(str(len(potential_malware)), ["bold", "red"])
                )
            )
            self.clean_files(potential_malware)
        return results
