from src.classifier import ImageClassifier
from src.constants import DATA_PATH, MODEL_PATH

classifier = ImageClassifier(DATA_PATH)
classifier.fit()
classifier.save(MODEL_PATH)
