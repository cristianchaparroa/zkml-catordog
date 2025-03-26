from content import ImageContent
from server import app
from fastapi import Depends, File, UploadFile
from classifier import Classifier
from supply import get_classifier, get_prover
from prover import ImageProver


@app.get("/", status_code=200)
def healthy():
    return "ok"


@app.post('/images/')
async def classify(
        file: UploadFile = File(...),
        classifier: Classifier = Depends(get_classifier),
        prover: ImageProver = Depends(get_prover)
):
    image_bytes = await file.read()
    content = ImageContent(image_bytes)
    result = await prover.generate_proof(content)
    return classifier.predict(content)
