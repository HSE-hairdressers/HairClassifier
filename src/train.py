from src.model.classifier import ImageClassifier
from src.utils.constants import DATA_PATH, MODEL_PATH

if __name__ == '__main__':
    classifier = ImageClassifier()
    classifier.fit(DATA_PATH)
    classifier.save(MODEL_PATH)
