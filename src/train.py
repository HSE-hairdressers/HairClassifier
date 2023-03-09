from src.classifier import ImageClassifier
from src.constants import DATA_PATH, MODEL_PATH

# DATA_PATH = "C:/Users/ageev/code/hair_dataset/haircuts3"
classifier = ImageClassifier(DATA_PATH)
classifier.fit()
classifier.save(MODEL_PATH)
# tf.keras.models.save_model(classifier.model, "C:/Users/ageev/code/hairstyle_classifier_nofaces")
