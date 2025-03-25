import tensorflow as tf
from tensorflow.keras.models import load_model


class Model:
    _model: tf.keras.Model

    def __init__(self, model_path: str):
        try:
            self._model = load_model(model_path)
        except Exception as e:
            print(e)

    def get_model(self) -> tf.keras.Model:
        return self._model
