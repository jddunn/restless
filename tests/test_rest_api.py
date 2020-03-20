import unittest
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

class MainAPITests(unittest.TestCase):

    def test_base_api_setup(self):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

if __name__ == '__main__':
    unittest.main()
