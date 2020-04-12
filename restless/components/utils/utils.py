import os
import subprocess

from .logger_utils import LoggerUtils
from .stats_utils import StatsUtils
from .stats_vis_utils import StatsVisUtils
from .misc_utils import MiscUtils

# from .db_caller import DB_Caller


logger = LoggerUtils()
stats = StatsUtils()
stats_vis = StatsVisUtils()
misc = MiscUtils()
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
        # self.print_logm("Initializing db: " + str(self.db.context))
        pass
