import datetime
import logging as logging


class Logger:
    """
    Logging with colors.
    """

    def __init__(self):
        # Add success level
        logging.SUCCESS = 25  # between WARNING and INFO
        logging.addLevelName(logging.SUCCESS, "SUCCESS")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("INFO")
        self.ch = logging.StreamHandler()  # console handler
        # Adjust formatting
        self.ch.setFormatter(CustomFormatter())
        self.logger.addHandler(self.ch)
        # Bind success method attribute to logger
        setattr(
            self.logger,
            "success",
            lambda message, *args: self.logger._log(logging.SUCCESS, message, args),
        )
        self.colored = colored  # Bind method to color text
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
        "reset": 0,
        "bold": 1,
        "faint": 2,
        "italic": 3,
        "underline": 4,
        "framed": 51,
        "encircled": 52,
        "overlined": 53,
        "slow_blink": 5,
        "rapid_blink": 6,
        "reversed": 7,
        "conceal": 8,
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "gray": 37,
        "d_gray": 90,
        "b_red": 91,
        "b_green": 92,
        "b_yellow": 93,
        "b_blue": 94,
        "b_magenta": 95,
        "b_cyan": 96,
        "white": 97,
        "bg_default": 49,
        "bg_black": 40,
        "bg_red": 41,
        "bg_green": 42,
        "bg_yellow": 43,
        "bg_blue": 44,
        "bg_magenta": 45,
        "bg_cyan": 46,
        "bg_gray": 47,
        "bg_d_gray": 100,
        "bg_l_red": 101,
        "bg_l_green": 102,
        "bg_l_yellow": 103,
        "bg_l_blue": 104,
        "bg_l_magenta": 105,
        "bg_l_cyan": 106,
        "bg_white": 107,
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
            "CRITICAL": "bg_red",
            "DEBUG": "bg_gray",
            "SUCCESS": "green",
        }
        clr = mapping.get(record.levelname)
        log_fmt = (
            colored("%(asctime)s", mapping.get("gray"))
            + "\t"
            + colored("(%(levelname)-4s)", clr)
            + "\t"
            + colored("%(message)s", mapping.get("white"))
            + "\t"
            + colored("(%(filename)s)", mapping.get("d_gray"))
        )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
