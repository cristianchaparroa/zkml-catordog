from content import ImageContent
from server import app
from fastapi import Depends, File, UploadFile
from classifier import Classifier
from supply import get_classifier


@app.get("/", status_code=200)
def healthy():
    return "ok"

@app.post('/images/')
async def classify(
        file: UploadFile = File(...),
        classifier: Classifier = Depends(get_classifier)
):
    image_bytes = await file.read()
    content = ImageContent(image_bytes)
    return classifier.predict(content)
