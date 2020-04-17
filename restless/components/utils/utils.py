import os
import subprocess

from .logger import Logger
from .stats import Stats
from .stats_vis import StatsVis
from .misc import Misc

# from .db_caller import DB_Caller

logger = Logger()
stats = Stats()
stats_vis = StatsVis()
misc = Misc()
# db = DB_Caller()


class Utils:
    """
    Various tools, including logging, database, and other high-level functions.
    """

    def __init__(self):
        self.logger = logger
        self.stats = stats
        self.stats_vis = stats_vis
        self.misc = misc
        # self.db = db
        pass
