# Make relative imports work for Docker
import sys
import os

PACKAGE_PARENT = "."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import uvicorn

from main import Restless

restless = Restless(run_system_scan=True)

SERVER_PORT = 4712

app = FastAPI(docs_url="/api_docs")

STATIC_ROOT = "/restless/docs"
# docs_path = os.path.abspath(os.path.join(STATIC_ROOT, 'docs'))
# print("THIS IS DOCS PATH: ", docs_path)

app.mount("/app_docs", StaticFiles(directory=STATIC_ROOT), name="app_docs")


@app.get("/")
def read_root():
    return ("Restless is running on port: ", SERVER_PORT)


if __name__ == "__main__":
    uvicorn.run(
        "server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info", reload=True
    )
