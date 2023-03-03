import tensorflow as tf
from src.classificator import HairClassificator

DATA_PATH = "onlyfaces/"
classificator = HairClassificator(DATA_PATH)
classificator.fit()
tf.keras.models.save_model(classificator.model, "hairstyle_classifier")
