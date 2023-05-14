from __future__ import print_function
import requests
import json
import cv2

# Test Client
def send_image(path: str):
    addr = 'http://localhost:8022'
    test_url = addr + '/api/test'
    # Specialize headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    img = cv2.imread(path)
    # Encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # Send http request and receive response
    response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)
    # decode response
    print(json.loads(response.text))

if __name__ == '__main__':
    while True:
        print("Enter image path or 'q' to quit:")
        str = input()
        if str == "q":
            break
        send_image(str)
