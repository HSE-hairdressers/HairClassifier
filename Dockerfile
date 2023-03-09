FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY onlyfaces onlyfaces
COPY hairstyle_classifier hairstyle_classifier
RUN pip3 install -r requirements.txt

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD [ "python3", "app.py"]