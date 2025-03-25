from response import ImageResult
from content import ImageContent
import tensorflow as tf # type: ignore

class Classifier:
    _model: tf.keras.Model

    def __init__(self):
        pass

    def predict(self, i: ImageContent) -> ImageResult:
        img_arr = i.get_data()
        prediction = self._model.predict(img_arr)[0][0]
        class_name = "dog" if prediction > 0.5 else "cat"
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)

        return ImageResult(prediction, confidence, class_name)
