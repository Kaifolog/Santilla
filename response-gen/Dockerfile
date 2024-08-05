# pull official base image
FROM python:3.9.19-slim-bullseye

# set work directory
WORKDIR /usr/src/response-gen

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN apt-get update
RUN pip install --upgrade pip
COPY ./response-gen ./response-gen
RUN pip install -r ./response-gen/requirements.txt
WORKDIR /usr/src/response-gen/response-gen
