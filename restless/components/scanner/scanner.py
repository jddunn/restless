from ..utils import utils

import sys
import os

sys.path.append("../")
sys.path.append("../../")

# Following lines are for assigning parent directory dynamically.
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from pe_analyzer import PEAnalyzer

pea = PEAnalyzer()
logger = utils.logger


class Scanner:
    """
    Extracts and analyzes information from files and classifies it for malware probability using NLP models.
    """

    def __init__(self):
        self.pea = pea
        logger.print_logm(
            "Initializing Restless.Scanner with PE Analyzer: " + str(self.pea)
        )
        pass

    def scan_full_system(self):
        results = []
        # results = self.pea.send_files_recursive("/home/ubuntu")
        print("This feature is currently being worked on!")
        return results

    def scan_folder(self, folderpath: str) -> list:
        results = self.pea.send_files_recursive(folderpath)
        return results

    def scan_file(self, filepath: str) -> list:
        results = self.scan_folder(filepath)
        return results
