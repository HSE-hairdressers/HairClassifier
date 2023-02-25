from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

from src.classificator import HairClassificator

DATA_PATH = "C:/Users/ageev/code/hair_dataset/onlyfaces/"

# Initialize the Flask application
app = Flask(__name__)
classificator = HairClassificator(DATA_PATH)
classificator.fit()
print("Classificator trained.")


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    result = classificator.predict(img)
    # build a response dict to send back to client
    response = {'message': f'image received. size={img.shape[1]}x{img.shape[0]}; result: {result}'
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
