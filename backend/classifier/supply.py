from fastapi import Depends
from classifier import Classifier
from server import app
import tensorflow as tf


def get_model() -> tf.keras.Model:
    return app.model


def get_classifier(model = Depends(get_model)):
    return Classifier(model)
