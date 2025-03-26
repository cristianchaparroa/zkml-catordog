from pydantic import BaseModel
from common.config import new_configuration
import os


class Config(BaseModel):
    srs_path: str
    settings_path: str
    verification_key_path: str
    proofs_dir_path: str


def new_verifier_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yml")
    return new_configuration(Config, config_path)
