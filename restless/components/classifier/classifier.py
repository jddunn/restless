import sys
import os
import time
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
    from restless.components.nlp import NLP
except Exception as e:
    from ..utils import utils as utils
    from ..nlp import NLP


logging = utils.logger
logger = utils.logger.logger
colored = logging.colored
same_line = logging.same_line
flush = logging.flush
misc = utils.misc

from multiprocessing import cpu_count


class Classifier:
    def __init__(self, load_default_hann_model=False):
        self.nlp = NLP(load_default_hann_model=load_default_hann_model)
        return

    def get_classification_results(
        self, fname: str, benign: float, malicious: float
    ) -> None:
        parent_path_to_fname = misc.get_parent_path_to_fname(fname)
        short_fname = fname.split("/")[len(fname.split("/")) - 1]
        classified_file_result = {
            "filename": fname,
            "benign": benign,
            "malicious": malicious,
            "time_scanned": misc.make_ts(),
        }

        # Colorizd percentages
        colored_fname = (
            colored(parent_path_to_fname, "gray")
            + colored("/", "bold")
            + " "
            + colored(short_fname, ["gray", "underline", "bold"])
        )
        colored_benign = misc.prob_to_percentage(benign)
        colored_malicious = misc.prob_to_percentage(malicious)

        clr_b = "d_gray"  # benign color(s)
        clr_m = "d_gray"  # malicious color(s)
        if benign > 0.3:
            clr_b = "yellow" if benign < 0.45 else ["yellow", "bold"]
        if benign >= 0.6:
            clr_b = ["green"] if benign < 0.8 else ["green", "bold"]
        if malicious > 0.15:
            clr_m = "yellow" if malicious < 0.25 else ["yellow", "bold"]
        if malicious >= 0.4:
            clr_m = "red" if malicious >= 0.6 and malicious <= 0.8 else ["red", "bold"]

        colored_benign = colored(colored_benign, clr_b)
        colored_malicious = colored(colored_malicious, clr_m)
        logger.info(
            "{} {} {} predicted: {} {} and {} {}.".format(
                colored("Scanned", "white"),
                colored_fname,
                colored("-", "d_gray"),
                colored_benign,
                colored("benign", clr_b),
                colored_malicious,
                colored("malicious", clr_m),
            )
        )
        return classified_file_result

    def analyze_scanned_files(
        self, file_results: list, default_malware_prob_threshold: float = 0.6
    ) -> tuple:

        all_results = []
        potential_malware_results = (
            []
        )  # Only results that pass default_malware_prob_threshold

        files_scanned = len(file_results)
        # Remove none from our results (meaning those files did not have any
        # extractable metadata for our classifier, for now at least)
        file_results = [res for res in file_results if res is not None]
        if not self._filter_initial_scan_results(
            files_scanned
        ):  # Checks to see if we have any files that can be classified
            return all_results, potential_malware_results
        count = (
            len(file_results) - 1 if len(file_results) - 1 > 0 else len(file_results)
        )
        logger.info(
            colored(
                "Sending {} files to the malware analysis / defense pipeline.".format(
                    colored(str(count), ["d_gray", "underline", "bold"]), "bold"
                )
            )
        )

        # Classification pipeline
        for file_result in file_results:
            fname = file_result[0]
            features = file_result[1]
            # Send features to NLP / HAN pipeline
            matrix_results = self.nlp.hann.build_feature_matrix_from_input_arr(features)
            result = (fname, self.nlp.hann.predict(matrix_results))
            benign = float(result[1][0])
            malicious = float(result[1][1])
            # Classify our results
            res = self.get_classification_results(fname, benign, malicious)
            if res["malicious"] >= default_malware_prob_threshold:
                potential_malware_results.append(res)
            all_results.append(res)

        flush(newline=True)
        logger.info(
            "\tRestless scanned a total of {} files, with {} sent to the malware classification / defense pipeline.".format(
                colored(files_scanned, ["d_gray", "bold", "underline"]),
                colored(len(all_results), ["gray", "bold", "underline"]),
            )
        )
        flush(newline=True)

        return all_results, potential_malware_results

    def _filter_initial_scan_results(self, files_scanned: int) -> bool:
        """Checks to see if we have any scanned files that can be analyzed."""
        if files_scanned == 0:
            logger.success(
                colored(
                    "Found no files that were scannable for malware (checked {} files).".format(
                        colored(str(files_scanned), ["bold", "underline"])
                    )
                )
                + colored(" The system seems to be safe.", ["bold", "green"])
            )
            return False
        else:
            return True
