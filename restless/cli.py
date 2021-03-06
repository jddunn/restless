from main import Restless
from components.utils import utils

import argparse

import os
import asyncio
import uvloop


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
        + " Enter '*' (in quotes) to run a full system scan.",
    )
    parser.add_argument(
        "-w",
        "--watch",
        required=False,
        help="If --watch is passed, Restless will always always watch the directory"
        + " (and all subdirs) passed directory. Enter '*' (in quotes) to watch"
        + " the home dir. New / modified files will be sent to the defense pipeline.",
    )
    args = parser.parse_args()
    uvloop.install()
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    event_loop = asyncio.get_event_loop()
    restless = Restless(run_system_scan=False)
    fp = args.input
    wfp = args.watch
    if fp and wfp:
        print(
            "Error! Please only pass in either an -i or -w (for input directory to scan now, or directory to watch and defend)."
        )
    if fp is None and wfp:
        try:
            event_loop.run_until_complete(restless.constant_watch(wfp))
        finally:
            event_loop.close()
        try:
            event_loop.run_until_complete(restless.scan_full_system())
        finally:
            event_loop.close()
    else:
        if fp is None and wfp is None:
            fp = "*"
        if fp is "*":
            try:
                event_loop.run_until_complete(restless.scan_full_system())
            finally:
                event_loop.close()
        else:
            if os.path.exists(fp):
                try:
                    event_loop.run_until_complete(restless.scan(fp))
                finally:
                    event_loop.close()
            else:
                print(
                    "Input is not a valid filepath! Please pass the absolute path or"
                    + " relative path if the files are inside the same dir."
                )
                exit()
