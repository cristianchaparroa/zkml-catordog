from server import app
from config import Config
from verifier import Verifier
from fastapi import Depends


def get_config() -> Config:
    return app.config


def get_verifier(conf=Depends(get_config)) -> Verifier:
    return Verifier(conf)
