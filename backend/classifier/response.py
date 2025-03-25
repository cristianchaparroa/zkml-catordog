class ImageResult:
    """Response model for prediction results"""
    _prediction: float
    _confidence: float
    _class_name: str

    def __init__(self, prediction: float, confidence: float, class_name: str):
        self._prediction = prediction
        self._confidence = confidence
        self._class_name = class_name
