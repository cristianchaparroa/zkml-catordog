from fastapi import FastAPI
from config import new_verifier_config

"""
All related to the server configuration should be defined here
"""
app = FastAPI(debug=True)


def load_config():
    conf = new_verifier_config()
    app.config = conf


@app.on_event("startup")
async def startup_event():
    load_config()
