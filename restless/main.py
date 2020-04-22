# Make relative imports work for Docker
import sys
import os
import time
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
from components.classifier import Classifier

misc = utils.misc
logging = utils.logger
logger = utils.logger.logger
colored = utils.logger.colored
flush = utils.logger.flush


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

        flush(newline=True)
        logger.info("\t" + colored("Restless initializing..", ["cyan", "bold"]))
        flush(newline=True)

        # Our default model will extract PE header data for classification
        self.classifier = Classifier(load_default_hann_model=True)
        self.scanner = Scanner()

        watch_pool = [os.path.abspath(path) for path in watch_pool]
        self.loop = asyncio.get_event_loop()  # reset event loop
        self.loop = asyncio.new_event_loop()
        self.watcher = Watcher(
            watch_pool, loop=self.loop, default_event_handler_cb=self.scan
        )

        if self.run_system_scan:
            self.loop.run_until_complete(self.scan_full_system())
        if constant_watch:
            self.loop.run_until_complete(self.constant_watch(watch_pool))

        return

    def constant_watch(self, watch_pool: list = ["*"]) -> None:
        """
        Constantly watches a list of directories for new / modified files,
        sending them to Restless's classification / defense pipeline.

        Args:
            watch_pool (list): List of directories or filepaths to
                constantly watch and scan.
        """
        self.watch_pool = watch_pool
        with ThreadPoolExecutor(max_workers=2) as executor:
            self.loop.run_until_complete(
                self.watcher.start_new_watch_thread(self.watch_pool)
            )
        return

    def quarantine_files(self, files: list) -> None:
        """Send potentially malicious files to quarantine for defense pipeline."""
        for file in files:
            pass
        return

    async def scan_full_system(self) -> list:
        """
        Starts a full system scan at the root path.

        Returns:
            list: List of results containing dictionaries with keys:
                "filename", "benign", "malicious", and "timestamp".
        """
        root = misc.get_os_root_path()
        results = await self.scan(root)
        return results

    async def scan(self, filepath: str, malware_prob_threshold: float = None):
        """
        Scans a file or directory recursively for malware.

        Args:
            filepath (str): File or directory to scan.
            malware_prob_threshold (float, optional): Probability threshold
                to classify something as malware. Defaults to .6.
        Returns:
            list: List of results containing dictionaries with keys:
                "filename", "benign", "malicious", and "timestamp".
        """
        start_time = time.time()

        if not malware_prob_threshold:
            malware_prob_threshold = self.default_malware_prob_threshold

        logger.info(
            "\t"
            + colored("Scanning", ["slow_blink", "bold"])
            + " system now at {}.".format(colored(filepath, "cyan"))
        )

        all_results = []
        potential_malware_results = []

        # Get features from our files to scan
        file_results = await self.scanner.scan_recursive(filepath)
        files_scanned = len(file_results)

        flush(newline=True)
        msg = "\t" + colored("Restless", ["bold", "slow_blink", "magenta", "underline"])
        msg += " " + colored(
            "defense pipeline working.", ["magenta", "bold", "slow_blink"]
        )
        logger.success(msg)
        flush(newline=True)

        # Send scanned files to classification pipeline
        (
            all_results,
            potential_malware_results,
        ) = await self.classifier.analyze_scanned_files(file_results)

        if len(potential_malware_results) > 0:
            logger.critical(
                colored(
                    "Found {} files to be potentially infected!".format(
                        colored(
                            str(len(potential_malware_results)),
                            ["bold", "red", "underline"],
                        )
                    ),
                    "red",
                )
            )
            self.quarantine_files(potential_malware_results)
        else:
            logger.success(
                colored(
                    "Scan finished sucessfully, found no potential malware!", "b_green"
                )
            )

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(
            "Restless scan took {} seconds.".format(
                colored(str(elapsed), ["d_gray", "bold"])
            )
        )

        return all_results, potential_malware_results
