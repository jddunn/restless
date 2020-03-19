import os
import subprocess

from .logger import Logger
from .db_caller import DB_Caller

logger = Logger()
db = DB_Caller()

DEFAULT_LOG_DIR = "./logs"

class Utils:
    """
    Various tools, including logging, database, and other high-level functions.
    """

    def __init__(self):
        self.logger = logger
        self.db = db
        pass

    def print_log(data={}):
        """
        Prints a log to sys output.

        Args:
          data (dict): Keys include: `level`, `text`. Timestamp will automatically be included.
        """
        self.logger._print_log(data)
        return

    def write_log(filepath=DEFAULT_LOG_DIR, data={}):
        """
        Prints and writes a log to disk.

        Args:
          filepath (str): Filepath of log to write to. Will default to latest log created in default directory.
          data (dict): Keys include: `level`, `text`. Timestamp will automatically be included.
        """
        return

    def check_if_in_docker_container(self):
        """
        Check to see if we're running inside a Docker container (via checking env var `APP_ENV`).
        """
        if os.environ.get("APP_ENV") == "docker":
            return True
        else:
            return False

    def get_list_of_most_recent_files(self, count: int = 1):
        """
        Returns list of most recent files modified / saved on disk.

        Args:
          count (int): Number of files to return (defaults to 1)

        Returns:
          list: List containing filenames in ascending order of modification time (oldest first).
        """
        count = str(ount)
        result = []
        return result

    def check_for_recent_filechanges(self, time_to_check: float = 60.0):
        """
        Checks to see if any files have been modified in the last time given (defaults to 60 seconds).

        Args:
          time_to_check (float): Number of seconds to go back in time to look

        Returns:
          bool: Whether any files have been changed or added to the system.
        """
        interval = str(interval)
        cmd = ["find", "~/", "-mtime", "-1", "-ls"]
        cmd = " ".join(cmd)
        res = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stderr, stdout = res.communicate()
        print(stderr, stdout)
        return

    def call_recentmost(self, count: int = 1):
        """
        Calls recentmost, which will sort the output of `find` using heapsort.
        """
        threshold = str(threshold)
        if self.check_if_in_docker_container():
            cmd = [
                "find",
                "~/",
                "-type",
                "f|./app/components/utils/recentmost",
                threshold,
            ]
        else:
            cmd = ["find", "~/", "-type", "f|./components/utils/recentmost", threshold]
        cmd = " ".join(cmd)
        res = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stderr, stdout = res.communicate()
        print(stderr, stdout)
        return
