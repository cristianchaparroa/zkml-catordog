from fastapi import FastAPI
from common.logger import *

"""
All related to the server configuration should be defined here
"""
app = FastAPI(debug=True)
