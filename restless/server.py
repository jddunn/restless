# Make relative imports work for Docker
import sys
import os
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from components.utils import Utils
from components.watcher import Watcher

from fastapi import FastAPI

import uvicorn

import restless

SERVER_PORT = 4712

app = FastAPI()

utils = Utils()

@app.get("/")
def read_root():
    return {"Restless is running on port: ", 4712}


print(utils.call_recentmost())

print(utils.check_for_recent_filechanges())

print(utils.check_if_in_docker_container())

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=4712, log_level="info", reload=True)

