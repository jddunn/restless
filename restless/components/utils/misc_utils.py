from datetime import datetime
from pathlib import Path


class MiscUtils:

    """Misc / general utilis."""

    def __init__(self):
        return

    @staticmethod
    def make_ts() -> str:
        """
        Makes human-readable timestamp.

        Returns:
            str: Timestamp with no microseconds.
        """

        return str(datetime.now().replace(microsecond=0))

    @staticmethod
    def check_if_in_docker_container() -> bool:
        """
        Check to see if we're running inside a Docker container (via checking env var `APP_ENV`).

        Returns:
            bool: Whether APP_ENV has been set to "Docker".
        """
        return True if os.environ.get("APP_ENV") == "docker" else False

    @staticmethod
    def check_if_child_in_parent(child_to_check: str, parent_to_check: str) -> bool:
        """
        Checks to see if dir / file is a child of another directory.

        Args:
            child_to_check (str): File or directory to check if child.
            parent_to_check (str): Directory to check if parent.
        Returns:
            bool: If file / dir is child of parent.
        """
        p_child = Path(child_to_check)
        p_parent = Path(parent_to_check)
        return True if p_parent in p_child.parents else False
