class ImageResult:
    """Response model for prediction results"""
    prediction: float
    confidence: float
    class_name: str

    def __init__(self, prediction: float, confidence: float, class_name: str):
        self.prediction = prediction
        self.confidence = confidence
        self.class_name = class_name
