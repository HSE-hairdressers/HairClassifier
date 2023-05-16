FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY hairstyle_classifier hairstyle_classifier
RUN pip3 install -r requirements.txt

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD [ "python3", "src/app.py"]