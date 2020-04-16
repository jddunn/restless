import os, sys
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import watchdog

# make dep imports work when running in dir and in outside scripts
PACKAGE_PARENT = "../../.."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
try:
    from restless.components.utils import utils as utils
except Exception as e:
    from ..utils import utils as utils

logger = utils.logger
misc = utils.misc


class Watcher:

    """
    Watcher constantly monitors the system scanning for newly updated or newly
    saved files, sending them to the classification / defense pipeline."
    """

    def __init__(self, watch_pool: list = None):
        self.watch_pool = watch_pool
        if self.watch_pool:
            logger.print_logm(
                "Restless.Watcher is now watching over system files in dirs: "
                + str(self.watch_pool)
                + "."
            )
            self.constant_scan(self.watch_pool, initializing=True)
        return

    async def constant_scan(self, dirs: list = None, initializing=False) -> list:
        """
        Main Watcher function.

        Args:
            dirs (list): List of directories to watch over. If $dirs == ["*"],
                then one watcher will be set over the root path of the machine.
            initializing (bool, optional): If true, don't perform check to see
                if we've already made watchers.

        Returns:
            list: Returns $self.watch_pool; list of directories being watched.
        """
        msg = ""
        if not dirs or dirs == ["*"]:
            msg = "Now watching over full system. Clearing all Watchers in pool."
            root = ""
            self.watch_pool = [root]
            logger.print_logm(msg)
        else:
            to_watch = []
            for fp in dirs:
                found = await (self.__check_if_already_watching_fp(fp))
                if found and not initializing:
                    msg = "{} is already being watched!".format(fp)
                    continue
                msg = "Now adding: {} to the Watcher pool.".format(fp)
                to_watch.append(fp)
                logger.print_logm(msg)
            self.watch_pool.extend(to_watch)
        logger.print_logm("Building watch pool..")
        __build_watch_pool(self.watch_pool)
        return self.watch_pool

    async def stop(self, dirs: list) -> None:
        """
        Stops watching over a list of directories. If passed a directory
        that isn't being watched, it will be skipped."""
        return

    async def __build__watch_pool(self, watch_pool: list) -> None:
        """Calls Watchgod."""
        return

    async def __check_if_already_watching_fp(self, fp: str) -> bool:
        """Checks to see if filepath is already being watched in watch_pool.
        """
        for to_watch in watch_pool:
            if misc.check_if_child_in_parent(fp, to_watch):
                return True
        return False

    async def __on_change_callback(self, fp: str) -> None:
        return


if __name__ == "__main__":
    watcher = Watcher()
    watcher.constant_scan(["*"])
