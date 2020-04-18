from main import Restless
from components.utils import utils

import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for using Restless to scan files for malware. Results of the"
        + "scan will be printed."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        help="File path to scan or folder path to scan recursively."
        + " Enter '*' to run a full system scan.",
    )
    parser.add_argument(
        "-w",
        "--watch",
        required=False,
        help="If --watch is passed, Restless will always always watch the input"
              + " directory (or the home dir by default), and send each new file added"
              + " / modified to the classification pipeline. "
    )
    args = parser.parse_args()
    fp = args.input
    if not fp:
        fp = "*"
    if fp is "*":
        restless = Restless(run_system_scan=False)
        # Run a full system scan
        restless.scan_full_system()
    else:
        if os.path.exists(fp):
            restless = Restless(run_system_scan=False)
            restless.scan(fp)
        else:
            print(
                "Input is not a valid filepath! Please pass the absolute path or" +
                " relative path if the files are inside the same dir."
            )
            exit()
