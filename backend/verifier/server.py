from fastapi import FastAPI
from config import new_verifier_config
from fastapi.middleware.cors import CORSMiddleware

"""
All related to the server configuration should be defined here
"""
app = FastAPI(debug=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def load_config():
    conf = new_verifier_config()
    app.config = conf


@app.on_event("startup")
async def startup_event():
    load_config()
