from main import Restless
from components.utils import utils

import argparse

import os

restless = Restless(run_system_scan=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CLI for using Restless to scan files for malware. Results of the scan will be printed.')
    parser.add_argument("-i", '--input', required=True,
                        help='File path to scan or folder path to scan recursively.'
                        )
    args = parser.parse_args()
    fp = args.input
    if fp is "*"
        # Run full system scan
    else:
       if os.path.exists(fp):
       else:
           print("Input is not a valid filepath! Please pass the absolute path or relative path if the files are inside the same dir.")
           exit()
