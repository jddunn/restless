class Logger:
    """
    Logger.
    """

    def __init__(self):
        pass

    def print_log(self, data: dict):
        if data["level"] is None:
            data["level"] = "info"
        print(data)

    def write_log(self, fp: str, data: dict):
        return
