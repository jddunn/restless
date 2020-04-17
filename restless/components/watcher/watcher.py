import os, sys
import asyncio
import uvloop
from watchdog.observers.polling import PollingObserver as Observer
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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

from events import AsyncFileClassifyEventHandler

logging = utils.logger
logger = utils.logger.logger
misc = utils.misc


class Watcher:

    """
    Watcher constantly monitors the system scanning for newly updated or newly
    saved files, sending them to the classification / defense pipeline."
    """

    def __init__(
        self, watch_pool: list, default_evt_handler=AsyncFileClassifyEventHandler
    ):
        self.watch_pool = watch_pool  # Array of paths to watch
        self.default_evt_handler = (
            default_evt_handler  # Event callback on watch modification signal
        )
        self.observer = None # Watchdog
        # if watched file changes
        if self.watch_pool:
            logger.info(
                "Restless.Watcher is now watching over system files in dirs: "
                + str(self.watch_pool)
                + "."
            )
            self.constant_scan(self.watch_pool, skip_check=True)
        return

    async def change_default_callback_evt(self, evt) -> None:
        """
        Changes the default event handler bound.
        """
        self.default_event_cb = evt
        return

    async def constant_scan(
        self, dirs: list = None, evt_handler: object = None, skip_check=False, time_interval:int=1
    ) -> list:
        """
        Main Watcher function.

        Args:
            dirs (list): List of directories to watch over. If $dirs == ["*"],
                then one watcher will be set over the root path of the machine.
            evt_handler (object): Event handler for each Watcher. Defaults to
                $self.default_evt_handler.
            skip_check (bool, optional): If true, don't perform check to see
                if we've already made watchers for the dirs. Defaults to false.

        Returns:
            list: Returns $self.watch_pool; list of directories being watched.
        """
        if not evt_handler:
            evt_handler = self.default_evt_handler
        msg = ""
        if not dirs or dirs == ["*"]:
            root = misc.get_os_root_path()
            msg = "Now watching over full system. Clearing all Watchers in pool."
            self.watch_pool = [root]
            logger.info(msg)
        else:
            to_watch = []
            for fp in dirs:
                if not skip_check:
                    found = await (self._check_if_already_watching_fp(fp))
                    if found:
                        msg = "{} is already being watched!".format(fp)
                        continue
                msg = "Now adding: {} to the Watcher pool.".format(fp)
                to_watch.append(fp)
                logger.info(msg)
            self.watch_pool.extend(to_watch)
        logger.info("Building watch pool..")
        await self._build_watch_pool(self.watch_pool, evt_handler)
        self.observer.start()
        try:
            while True:
                time.sleep(time_interval)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()
        return self.watch_pool

    async def check_if_already_watching_fp(self, fp: str) -> bool:
        """Checks to see if filepath is already being watched in watch_pool.
        """
        for to_watch in watch_pool:
            if misc.check_if_child_in_parent(fp, to_watch):
                return True
        return False

    async def stop(self, dirs: list) -> None:
        """
        Stops watching over a list of directories. If passed a directory
        that isn't being watched, it will be skipped."""
        return

    async def _build_watch_pool(self, watch_pool: list, evt_handler = None) -> None:
        """Calls Watchdog."""
        self.observer = Observer()
        if not evt_handler:
            evt_handler = self.default_evt_handler
        for watched_dir in watch_pool:
            self.observer.schedule(evt_handler, watched_dir, recursive=True)
        return

    async def main(self, arg):
        res = await watcher.constant_scan(arg)
        print(res)
        return res


if __name__ == "__main__":
    watcher = Watcher([])
    uvloop.install()
    asyncio.run(watcher.main(["*"]))
