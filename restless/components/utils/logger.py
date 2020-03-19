class Logger:
    """
    Logger component. Private methods will be called by higher-level `Utils`.
    """

    def __init__(self):
        pass

    def _print_log(self, data: dict):
        if data["level"] is None:
            data["level"] = "info"
        print(data)

    def _write_log(self, fp: str, data: dict):
        return
