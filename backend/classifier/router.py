from server import app

@app.get("/", status_code=200)
def healthy():
    return "ok"
