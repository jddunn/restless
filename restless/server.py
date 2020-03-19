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

app = FastAPI()

docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))

print(os.path.dirname(__file__))
print("THIS IS DOCS_PATH: ", docs_path)

app.mount(docs_path, StaticFiles(directory=docs_path), name="docs")

@app.get("/")
def read_root():
    return("Restless is running on port: ", SERVER_PORT)

if __name__ == "__main__":
    uvicorn.run(
        "server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info", reload=True
    )
