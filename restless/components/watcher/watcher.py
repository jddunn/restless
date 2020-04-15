from ..utils import utils
from watchgod import watch

logger = utils.logger
misc = utils.misc


class Watcher:

    """
    Watcher constantly monitors the system scanning for newly updated or newly
    saved files, sending them to the classification / defense pipeline."
    """

    def __init__(self, dirs_to_watch: list = None):
        self.dirs_to_watch = dirs_to_watch
        if self.dirs_to_watch:
            logger.print_logm(
                "Restless.Watcher is now watching over system files in dirs: "
                + str(self.dirs_to_watch)
                + "."
            )
        self.watch_pool = []
        return

    def constant_scan(self, dirs: list = None) -> None:
        """
        Main Watcher function.

        Args:
            dirs (list): List of directories to watch over

        Returns:
            list: List of filenames (with absolute paths) that have been
                  updated or saved since the last `scan` was performed
                  (in descending order of modification timestamps).
        """
        msg = ""
        if not dirs or dirs == ["*"]:
            msg = "Now watching over full system. Clearing all Watchers in pool."
            root = ""
            self.watch_pool = [root]
        else:
            msg = "Now adding: {} to the Watcher pool.".format(dirs)
        logger.print_logm(msg)
        return

    def stop(self, dirs: list) -> None:
        """
        Stops watching over a list of directories. If passed a directory
        that isn't being watched, it will be skipped."""
        return

    def __check_if_already_watching_fp(self, fp: string) -> bool:
        for to_watch in watch_pool:
            if misc.check_if_child_in_parent(fp, to_watch):
                return True
        return False
