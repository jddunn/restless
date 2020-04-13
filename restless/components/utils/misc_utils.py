from datetime import datetime


class MiscUtils:

    """Misc / general utilis."""

    def __init__(self):
        return

    def make_ts(self) -> str:
        return str(datetime.now().replace(microsecond=0))

    def check_if_in_docker_container(self) -> bool:
        """
        Check to see if we're running inside a Docker container (via checking env var `APP_ENV`).
        """
        if os.environ.get("APP_ENV") == "docker":
            return True
        else:
            return False
