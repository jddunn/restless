from components.utils import Utils
from components.watcher import Watcher
from components.scanner import Scanner

utils = Utils()
watcher = Watcher()
scanner = Scanner()


class Restless:
    """
    Main Restless module.
    """

    def __init__(self):
        print("I'm restless")
