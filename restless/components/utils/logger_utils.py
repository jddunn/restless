import datetime
import logging as logging


class LoggerUtils:
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

    def print_logm(self, message: str) -> None:
        """
        Easy print log.

        Args:
          message (str): Message to print to `INFO` level.
        """
        self.print_log({"message": message})
        return

    def print_log(self, data: dict):
        """
        Prints a log to sys output.

        Args:
          data (dict): Keys include: `level`, `message`. Timestamp will automatically be included.
                       `Level` will default to `INFO`.
        """
        level = data.get("level")
        if level is None:
            level = "INFO"
        if level == "info":
            self.logging.info(data["message"])
        elif level == "critical":
            self.logging.critical(data["message"])
        elif level == "error":
            self.logging.error(data["message"])
        elif level == "warning":
            self.logging.warning(data["message"])
        elif level =="debug":
            self.logging.debug(data["message"])
        else:
            self.logging.info(data["message"])
        return

    def write_log(self, data: dict, filepath: str) -> bool:
        """
        Prints and writes a log to disk.

        Args:
          filepath (str): Filepath of log to write to. Will default to latest log created in default directory.
          data (dict): Keys include: `level`, `text`. Timestamp will automatically be included.
        """
        return
