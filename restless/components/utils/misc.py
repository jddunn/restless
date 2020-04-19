import os, sys
from datetime import datetime
from pathlib import Path
import pickle


class Misc:

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

    @staticmethod
    def get_os_root_path() -> list:
        """
        Returns root / system paths of current machine.

        Returns:
            list: Root / important system paths to scan.
        """
        return os.path.expanduser("~")

    @staticmethod
    def read_pickle_data(path):
        """Reads pickled data."""
        if os.path.isfile(path):
            with open(path, "rb") as f:
                try:
                    return pickle.load(f)
                except Exception as e:
                    print("Could not read pickle!")
                    print(e)
                    return None
        else:
            return None

    @staticmethod
    def write_pickle_data(data: object, path: str, overwrite: bool=True):
        """
        Writes pickled data to disk (overwrites existing paths by default).
        """
        if not overwrite:
            if os.path.isfile(path):
                print("{} exists and will not be overwritten!")
                return None
        try:
            with open(path, "wb") as f:
                try:
                    pickle.dump(data, f)
                except Exception as e:
                    return None
        except:
            return None
        return path
