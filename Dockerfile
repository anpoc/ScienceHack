FROM python:3.10.18-bullseye

RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt