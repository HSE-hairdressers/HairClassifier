from flask import Flask, request, Response
import jsonpickle
import numpy as np
import logging
import cv2
from waitress import serve

from src.constants import MODEL_PATH
from src.restoredmodel import RestoredModel


# Initialize the Flask application
app = Flask(__name__)
classifier = RestoredModel(MODEL_PATH)


# Route http post method
@app.route('/api/test', methods=['POST'])
def classify_image():
    '''
    Handle function, which catches the image
    :return: response with the most probable class
    '''
    logging.debug('Python Server caught request.')
    # Convert string of image data to uint8
    arr = np.fromstring(request.data, np.uint8)
    # Decode image
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    logging.debug('Image decoded successfully')
    # Preprocess image and predict its class
    try:
        result = classifier.predict(img)
        message = "Hairstyle classified"
    except ValueError:
        result = 0
        message = "Face not detected"
    logging.debug('Finished classification')
    # Build a response dict
    response = {'message': message,
                'size': {
                    'width': img.shape[1],
                    'height': img.shape[0]
                },
                'result': f'{result}'}
    logging.debug('Sending result')
    # Encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# Start flask application
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    serve(app, host="localhost", port=8022)
