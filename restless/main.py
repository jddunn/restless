# Make relative imports work for Docker
import sys
import os
import asyncio
import uvloop

from concurrent.futures import ThreadPoolExecutor

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
colored = utils.logger.colored


class Restless(object):
    """
    Main Restless module.
    """

    def __init__(
        self,
        run_system_scan: bool = False,  # Run full system scan (from home dir)
        constant_watch: bool = False,  # Constantly defend system (defaults to home dir)
        # by scanning and cleaning new / modified files
        watch_pool: list = ["*"],  # List of dirs to constantly watch / defend
        # "*" will make it default to home dir
        default_malware_prob_threshold=0.6,  # Prob threshold to classify as malware
    ):
        uvloop.install()  # make event loop fast
        self.run_system_scan = run_system_scan
        self.default_malware_prob_threshold = default_malware_prob_threshold
        logger.info("Restless initializing..")
        self.scanner = Scanner()
        watch_pool = [os.path.abspath(path) for path in watch_pool]
        self.event_loop = asyncio.get_event_loop()  # reset event loop
        self.event_loop = asyncio.new_event_loop()
        self.watcher = Watcher(watch_pool, loop=self.event_loop, default_event_handler_cb=self.scan)
        # Our default model will extract PE header data for classification
        self.nlp = NLP(load_default_hann_model=True)
        if self.run_system_scan:
            self.scan_full_system()
        if constant_watch:
            self.constant_watch(watch_pool)
        return

    def clean_files(self, infected_files: list) -> None:
        for file in infected_files:
            pass
        return

    def constant_watch(self, watch_pool: list = ["*"]) -> None:
        self.watch_pool = watch_pool
        self.event_loop = asyncio.get_event_loop()  # reset event loop
        self.event_loop = asyncio.new_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            self.event_loop.run_until_complete(
                self.watcher.start_new_watch_thread(
                    self.watch_pool
                )
            )
        return

    def scan_full_system(self):
        root = misc.get_os_root_path()
        results = self.scan(root)
        return results

    async def scan(self, filepath: str, malware_prob_threshold: float = None):
        if not malware_prob_threshold:
            malware_prob_threshold = self.default_malware_prob_threshold
        logger.info(colored("Scanning", "slow_blink") + " system now at {}.".format(colored(filepath, "cyan")))
        results = []
        potential_malware = []
        file_results = await self.scanner.scan_recursive(filepath)
        files_scanned = len(file_results)
        # Remove none from our results (meaning those files did not have any
        # extractable metadata for our classifier, for now at least)
        file_results = [res for res in file_results if res]
        if len(file_results) == 0:
            logger.success(
                colored(
                    "Found no files that were scannable for malware (checked {} files).".format(
                        colored(str(files_scanned), "bold"), "green"
                    )
                )
            )
            logger.success(colored("The system appears to be safe.", "green"))
            return
        for file_result in file_results:
            fname = file_result[0]
            path_to_fname = fname.split("/")
            path_to_fname.pop()
            path_to_fname = "/".join(path_to_fname)
            short_fname = fname.split("/")[len(fname.split("/")) - 1]
            features = file_result[1]
            if len(self.nlp.hann.features) > 0:
                features = [x for x in features if x in self.nlp.hann.features]
            matrix_results = self.nlp.hann.build_feature_matrix_from_input_arr(features)
            result = (fname, self.nlp.hann.predict(matrix_results))
            results.append(result)
            colored_fname = (
                colored(path_to_fname, "gray")
                + colored("/", "bold")
                + colored(short_fname, ["gray", "underline"])
            )
            benign = float(result[1][0])
            malicious = float(result[1][1])
            # Colorize percentages
            colored_benign = colored(benign, "d_gray") + colored("%", "d_gray")
            colored_malicious = colored(malicious, "d_gray") + colored("%", "d_gray")
            if benign > 0.6:
                clr = "green" if benign < 0.8 else "b_green"
                colored_benign = colored(benign, clr)
                colored_benign += colored("%", clr)
            if malicious > 0.1 and malicious < 0.4:
                clr = "yellow"
                colored_malicious = colored(malicious, clr)
                colored_malicious += colored("%", clr)
            if malicious > 0.6:
                clr = "red" if malicious > 0.8 else "b_red"
                colored_malicious = colored(malicious, clr)
                colored_malicious += colored("%", clr)
            if malicious >= malware_prob_threshold:
                potential_malware.append(fname)
            logger.info(
                "{} {} {} predicted: {} {} and {} {}.".format(
                    colored("Scanned", "white"),
                    colored_fname,
                    colored("-", "d_gray"),
                    colored_benign,
                    colored("benign", "gray"),
                    colored_malicious,
                    colored("malicious", "gray"),
                )
            )
        if len(potential_malware) > 0:
            logger.critical(
                "Found {} files to be potentially infected!".format(
                    colored(str(len(potential_malware)), ["bold", "red"])
                )
            )
            self.clean_files(potential_malware)
        else:
            logger.success(
                colored(
                    "Scan finished sucessfully, found no potential malware!", "b_green"
                )
            )
        return results
