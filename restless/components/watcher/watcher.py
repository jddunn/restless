import os, sys
import asyncio
import uvloop
import time as time
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from _thread import start_new_thread

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

uvloop.install()


class Watcher:

    """
    Watcher constantly monitors the system scanning for newly updated or newly
    saved files, sending them to the classification / defense pipeline."
    """

    def __init__(
        self, watch_pool: list, default_evt_handler=AsyncFileClassifyEventHandler()
    ):
        self.watch_pool = watch_pool  # Array of paths to watch
        self.default_evt_handler = (
            default_evt_handler  # Event callback on watch modification signal
        )
        self.watchdog = AIOWatchdog(
            self.watch_pool, event_handler=self.default_evt_handler
        )
        self.watching = True  # As long as this is true, Watcher will be watching
        return

    async def start_new_watch_thread(
        self, loop, executor=None, dirs: list = None
    ) -> None:
        try:
            result = start_new_thread(self.constant_watch, (dirs,))
        finally:
            pass
        return result

    async def change_default_callback_evt(self, evt) -> None:
        """
        Changes the default event handler bound.
        """
        self.default_event_handler = evt
        return

    def constant_watch(
        self,
        dirs: list = None,
        evt_handler: object = None,
        skip_check=False,
        time_interval: int = 3,
    ) -> None:
        """
        Main Watcher function. Requires an executor to run in a separate thread.

        Args:
            dirs (list): List of directories to watch over. If $dirs == ["*"],
                then one watcher will be set over the root path of the machine.
            evt_handler (object): Event handler for each Watcher. Defaults to
                $self.default_evt_handler.
            skip_check (bool, optional): If true, don't perform check to see
                if we've already made watchers for the dirs. Defaults to false.
        """
        self.watch_pool = []
        if not evt_handler:
            evt_handler = self.default_evt_handler
        msg = ""
        if not dirs or dirs == ["*"] or dirs == ([],):
            msg = (
                logging.colored("Restless", "bold")
                + " is now "
                + logging.colored("watching over", "slow_blink")
                + " the full system."
            )
            root = misc.get_os_root_path()
            self.watch_pool = [root]
        else:
            to_watch = []
            for fp in dirs:
                if not skip_check:
                    found = self.check_if_already_watching_fp(fp, self.watch_pool)
                    if found:
                        msg = "{} is already being watched!".format(fp)
                        continue
                msg = "Adding: {} to the Watcher pool.".format(fp)
                to_watch.append(fp)
                logger.info(msg)
            self.watch_pool.extend(to_watch)
            msg = (
                logging.colored("Restless", "bold")
                + " is now "
                + logging.colored("watching over", "slow_blink")
                + " the system."
            )
        self.watchdog = AIOWatchdog(
            self.watch_pool, event_handler=self.default_evt_handler, recursive=True
        )
        logger.info(msg)
        self.watchdog.start()
        try:
            while True:
                time.sleep(time_interval)
        except KeyboardInterrupt:
            self.watchdog.stop()

    def check_if_already_watching_fp(self, fp: str, watch_pool) -> bool:
        """Checks to see if filepath is already being watched in watch_pool.
        """
        for to_watch in watch_pool:
            if misc.check_if_child_in_parent(fp, to_watch):
                return True
        return False

    def keep_loop(self, time_interval: int = 3) -> None:
        """Keeps Watcher alive on an interval.
           Should be called in separate thread.
        """
        if self.watching:
            time.sleep(time_interval)
            self.keep_loop(time_interval)
        else:
            self.watchdog.stop()
            return

    def stop(self, dirs: list) -> None:
        """
        Stops watching over a list of directories. If passed a directory
        that isn't being watched, it will be skipped."""
        return


# Asyncio Watchdog wrapper code taken from
# https://github.com/biesnecker/hachiko, but modified
# to work with multiple watched directories
class AIOWatchdog(object):
    def __init__(self, path=".", recursive=True, event_handler=None, observer=None):
        if observer is None:
            self._observer = Observer()
        else:
            self._observer = observer

        evh = event_handler or AIOEventHandler()

        if isinstance(path, list):
            for _path in path:
                self._observer.schedule(evh, _path, recursive)
        else:
            self._observer.schedule(evh, path, recursive)

    def start(self):
        self._observer.start()

    def stop(self):
        self._observer.stop()
        self._observer.join()


if __name__ == "__main__":
    dirs = ["/home/ubuntu"]
    watcher = Watcher(dirs)
    event_loop = asyncio.get_event_loop()
    event_loop = asyncio.new_event_loop()
    with ThreadPoolExecutor(max_workers=4) as executor:
        event_loop.run_until_complete(
            watcher.start_new_watch_thread(event_loop, executor, dirs)
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        event_loop.run_until_complete(watcher.keep_loop())
