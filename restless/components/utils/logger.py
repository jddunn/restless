import datetime
import logging


class Logger:
    """
    Logger component. Private methods will be called by higher-level `Utils`.
    """

    def __init__(self):
        self.logging = logging
        self.logging.basicConfig(
            format="%(asctime)s %(levelname)-4s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        pass

    def _print_log(self, data: dict):
        level = data.get("level")
        if level is None:
            level = "INFO"
        if level is "INFO":
            self.logging.info(data["message"])
        return

    def _write_log(self, fp: str, data: dict):
        return
