from ..utils import Utils

utils = Utils()

print(utils.call_recentmost())

print(utils.check_for_recent_filechanges())

print(utils.check_if_in_docker_container())


class Watcher:

    """
    Watcher constantly monitors the system scanning for newly updated or newly saved files, sending them to the classification / defense pipeline."
    """

    def __init__(self):
        print(
            "Restless.Watcher is now watching over system and scanning new incoming files."
        )
        pass

    def constant_scan(self):
        return
