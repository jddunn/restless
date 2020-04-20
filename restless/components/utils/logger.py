import datetime
import logging as logging
import sys

# Format logs with extended metadata like functions, files, lineno
extended_log_data = False

import time


class Logger:
    """
    Logging with colors.
    """

    level = "INFO"

    def __init__(self):
        # Add success level
        logging.SUCCESS = 25  # between WARNING and INFO
        logging.addLevelName(logging.SUCCESS, "SUCCESS")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.level)
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
        self.same_line = same_line
        return

    def flush(self, newline=False) -> None:
        CURSOR_UP_ONE = "\033[K"
        ERASE_LINE = "\x1b[2K"
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE + "\r")
        if newline:
            sys.stdout.write(ERASE_LINE + "\n")
        return

    def change_logging_config(self, config: dict) -> None:
        """Change logging configuration (uses Python's logging module)."""
        logging.basicConfig(config)
        return

    def change_logging_level(self, level: str) -> None:
        """Changes level of logger and console handler."""
        self.level = level
        if level == "debug":
            self.logger.setLevel(logging.DEBUG)
            self.ch.setLevel(logging.DEBUG)
        elif level == "info":
            self.logger.setLevel(logging.INFO)
            self.ch.setLevel(logging.INFO)
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
            raise ValueError("Invalid logging level to set!")


class ANSIColor:
    """
    Utility to return ansi colored text.
    Modified from
    https://gist.github.com/hit9/5635505
    with full styles and multi-style functionality.
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

    def same_line(self, text: str):
        text = "\033[K" + "\033[F" + "\033[K" + text
        return text

    def colored(self, text: str, color=None):
        if isinstance(color, list):
            # Apply multiple styles
            result = ""
            for _color in color:
                clr = self.colors[_color]
                result += (self.prefix + "%d" + "m") % clr
            result += ("%s") % text
            result += self.suffix
            return result
        else:
            if color not in self.colors:
                color = "white"
            clr = self.colors[color]
            return (self.prefix + "%dm%s" + self.suffix) % (clr, text)


colored = ANSIColor().colored
same_line = ANSIColor().same_line


class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colors and count warning / errors
    https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """

    mapping = {
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bg_red",
        "DEBUG": "bg_gray",
        "SUCCESS": "green",
    }

    def format(self, record):
        message = record.getMessage()
        clr = self.mapping.get(record.levelname)
        if extended_log_data == True:
            # Format log with extended props for debugging
            log_fmt = (
                colored("%(asctime)-4s", "gray")
                + "\t"
                + colored("(%(levelname)-4s)", clr)
                + "\t"
                + colored("%(message)-4s", "white")
                + "\t"
                + colored("(%(filename)s : ", "d_gray")
                + colored("line %(lineno)s in function ", "d_gray")
                + colored("%(funcName)s)", "d_gray")
            )
            formatter = logging.Formatter(log_fmt)
        else:
            log_fmt = (
                colored("%(asctime)-4s", self.mapping.get("gray"))
                + "\t"
                + colored("(%(levelname)-4s)", clr)
                + "\t"
                + colored("%(message)s", self.mapping.get("white"))
            )
            formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)
