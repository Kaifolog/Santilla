# pull official base image
FROM python:3.9.19-slim-bullseye

# set work directory
WORKDIR /usr/src/classifier

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN apt-get update
RUN pip install --upgrade pip
COPY ./classifier ./classifier
RUN pip install -r ./classifier/requirements.txt
WORKDIR /usr/src/classifier/classifier
