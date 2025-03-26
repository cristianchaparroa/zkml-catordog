import uvicorn
import handler

if __name__ == "__main__":
    uvicorn.run(handler.app, host="0.0.0.0", port=8000)
