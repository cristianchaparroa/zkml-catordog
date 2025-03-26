import os

from fastapi import FastAPI
from common.logger import *
from model import Model
import subprocess

"""
All related to the server configuration should be defined here
"""
app = FastAPI(debug=True)


def get_git_root():
    """Get the root directory of the git repository."""
    try:
        git_root = (subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT)
                    .decode().strip())
        return git_root
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to the current directory if not in a git repo or git not installed
        return os.path.dirname(os.path.abspath(__file__))


def get_model_path() -> str:
    return os.path.join(get_git_root(), "classifier", "tiny_model.keras")


def load_model():
    model_path = os.environ.get("MODEL_PATH", get_model_path())
    app.model = Model(model_path).get_model()


@app.on_event("startup")
async def startup_event():
    load_model()
