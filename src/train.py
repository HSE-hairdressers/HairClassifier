import tensorflow as tf
from src.classificator import HairClassificator

DATA_PATH = "C:/Users/ageev/code/hair_dataset/onlyfaces/"
classificator = HairClassificator(DATA_PATH)
classificator.fit()
tf.keras.models.save_model(classificator.model, "C:/Users/ageev/code/hairstyle_classifier")
