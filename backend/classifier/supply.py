from fastapi import Depends
from classifier import Classifier
from server import app
from config import Config
import tensorflow as tf
from prover import ImageProver


def get_config() -> Config:
    return app.config


def get_model() -> tf.keras.Model:
    return app.model


def get_classifier(model=Depends(get_model)):
    return Classifier(model)


def get_prover(conf=Depends(get_config)):
    return ImageProver(conf)
