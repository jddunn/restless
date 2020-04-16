import datetime
import logging as logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

class LoggerUtils:
    """
    Logger component. Private methods will be called by higher-level `Utils`.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.ch = logging.StreamHandler() # console handler
        self.ch.setLevel(logging.INFO)
        # Adjust formatting
        self.ch.setFormatter(CustomFormatter())
        self.logger.addHandler(self.ch)
        return

    def change_logging_config(self, config: dict) -> None:
        """Change logging configuration (uses Python's logging module)."""
        logging.basicConfig(config)
        return

    def change_logging_level(self, level: str) -> None:
        """Changes level of logger and console handler."""
        if level == "debug":
            self.logging.setLevel(logging.DEBUG)
            self.ch.setLevel(logging.DEBUG)
        elif level == "info":
            self.logging.setLevel(logging.DEBUG)
            self.ch.setLevel(logging.DEBUG)
        elif level == "warning":
            self.logging.setLevel(logging.WARNING)
            self.ch.setLevel(logging.WARNING)
        elif level == "error":
            self.logging.setLevel(logging.ERROR)
            self.ch.setLevel(logging.ERROR)
        elif level == "criticial":
            self.logging.setLevel(logging.CRITICAL)
            self.ch.setLevel(logging.CRITICAL)
        else:
            raise ValueError("Invalid logging level!")


    def print_logm(self, message: str) -> None:
        """
        Easy print log to info level with timestamp.

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
        elif level == "debug":
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


class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
