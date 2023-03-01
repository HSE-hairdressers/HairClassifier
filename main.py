from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
from tensorflow import keras

from src.classifier import HairClassifier
DATA_PATH = "C:/Users/ageev/code/hair_dataset/onlyfaces/"

# Initialize the Flask application
app = Flask(__name__)
classifier = HairClassifier(keras.models.load_model("C:/Users/ageev/code/hairstyle_classifier"), DATA_PATH)
# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    print('Python Server caught request.')
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print('Image decoded successfully')
    try:
        result = classifier.predict(img)
        message = "Hairstyle classified"
    except ValueError:
        result = 0
        message = "Face not detected"
    print('Finished classification')
    # build a response dict to send back to client
    response = {'message': message,
                'size': {
                    'width': img.shape[1],
                    'height': img.shape[0]
                },
                'result': f'{result}'}
    print('Sending result')
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
