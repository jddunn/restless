from ..utils import utils


class Watcher:

    """
    Watcher constantly monitors the system scanning for newly updated or newly saved files,
    sending them to the classification / defense pipeline."
    """

    def __init__(self):
        utils.print_logm(
            "Restless.Watcher is now watching over system and scanning new incoming files."
        )
        pass

    def constant_scan(self):
        """
        Main Watcher function.

        Returns:
            list: List of filenames (with absolute paths) that have been updated or saved
                  since the last `scan` was performed (in descending order of modification
                  timestamps).
        """
        return

    def check_for_new_data(self):
        """
        Checks to see if there's any new incoming data saved or updated to the system
        (verified by file modification timestamps).
        """
        return
