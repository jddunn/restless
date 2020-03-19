# Make relative imports work for Docker
import sys
import os

PACKAGE_PARENT = "."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from fastapi import FastAPI

import uvicorn

from main import Restless

restless = Restless(run_system_scan=True)

SERVER_PORT = 4712

app = FastAPI()


@app.get("/")
def read_root():
    return("Restless is running on port: ", SERVER_PORT)


if __name__ == "__main__":
    uvicorn.run(
        "server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info", reload=True
    )
