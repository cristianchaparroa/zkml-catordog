from fastapi import Path, Depends
from server import app
from verifier import Verifier
from supply import get_verifier


@app.get("/", status_code=200)
def healthy():
    return "ok"


@app.get("/verifies/{id}")
async def verify_id(
        id: str = Path(..., description="The ID to verify"),
        verifier: Verifier = Depends(get_verifier)
):
    return verifier.verify(id)
