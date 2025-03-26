class ImageResult:
    """Response model for prediction results"""
    id: str
    prediction: float
    confidence: float
    class_name: str

    def __init__(self, id: str, prediction: float, confidence: float, class_name: str):
        self.id = id
        self.prediction = prediction
        self.confidence = confidence
        self.class_name = class_name
