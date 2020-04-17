import datetime
import logging as logging


class Logger:
    """
    Logging with colors.
    """

    def __init__(self):
        self.levels = [
            "debug",
            "info",
            "success",
            "warning",
            "level",
            "error",
            "critical",
        ]
        # logging.basicConfig(level=logging.INFO)
        # Add success level
        logging.SUCCESS = 25  # between WARNING and INFO
        logging.addLevelName(logging.SUCCESS, "SUCCESS")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()  # console handler
        # self.ch.setLevel(logging.INFO)
        # Adjust formatting
        self.ch.setFormatter(CustomFormatter())
        self.logger.addHandler(self.ch)
        # Bind success method attribute to logger
        setattr(
            self.logger,
            "success",
            lambda message, *args: self.logger._log(logging.SUCCESS, message, args),
        )
        return

    def print_logm(self, text: str, save: bool = False) -> None:
        """
        Easy print log to info level with timestamp.
        Meant to be more high-level than Python's logging.

        Args:
          text (str): Message to print to `INFO` level.
        """
        self.print_log({"text": text})
        return

    def print_log(self, data: dict, save: bool = False):
        """
        Prints a log to sys output.
        Meant to be more high-level than Python's logging.

        Args:
          data (dict): Keys include: `level`, `text`. Timestamp will automatically be included.
                       `Level` will default to `INFO`.
        """
        level = data.get("level")
        if level is None:
            level = "info"
        if level == "info":
            self.logger.info(str(data["text"]))
        elif level == "critical":
            self.logger.critical(data["text"])
        elif level == "error":
            self.logger.error(data["text"])
        elif level == "warning":
            self.logger.warning(data["text"])
        elif level == "success":
            self.logger.success(data["text"])
        elif level == "debug":
            self.logger.debug(data["text"])
        else:
            self.logger.info(data["text"])
        return

    def change_logging_config(self, config: dict) -> None:
        """Change logging configuration (uses Python's logging module)."""
        logging.basicConfig(config)
        return

    def change_logging_level(self, level: str) -> None:
        """Changes level of logger and console handler."""
        if level == "debug":
            self.logger.setLevel(logging.DEBUG)
            self.ch.setLevel(logging.DEBUG)
        elif level == "info":
            self.logger.setLevel(logging.DEBUG)
            self.ch.setLevel(logging.DEBUG)
        elif level == "warning":
            self.logger.setLevel(logging.WARNING)
            self.ch.setLevel(logging.WARNING)
        elif level == "error":
            self.logger.setLevel(logging.ERROR)
            self.ch.setLevel(logging.ERROR)
        elif level == "criticial":
            self.logger.setLevel(logging.CRITICAL)
            self.ch.setLevel(logging.CRITICAL)
        else:
            raise ValueError("Invalid logging level!")


class ANSIColor:
    """
    Utility to return ansi colored text.
    https://gist.github.com/hit9/5635505
    """

    colors = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "bgred": 41,
        "bggrey": 100,
    }
    prefix = "\033["
    suffix = "\033[0m"

    def colored(self, text, color=None):
        if color not in self.colors:
            color = "white"
        clr = self.colors[color]
        return (self.prefix + "%dm%s" + self.suffix) % (clr, text)


colored = ANSIColor().colored


class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """

    def format(self, record):

        message = record.getMessage()

        mapping = {
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bgred",
            "DEBUG": "bggrey",
            "SUCCESS": "green",
        }
        clr = mapping.get(record.levelname)
        log_fmt = colored("%(asctime)s", mapping.get("white")) + "\t" + colored("(%(levelname)-4s)", clr) + "\t" + colored("%(message)s", mapping.get("white")) + "\t" +  colored('(%(name)s)', mapping.get('bggrey'))
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
