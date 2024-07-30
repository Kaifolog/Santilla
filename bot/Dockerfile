# pull official base image
FROM python:3.9.19-slim-bullseye

# set work directory
WORKDIR /usr/src/bot

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN apt-get update
RUN pip install --upgrade pip
COPY ./bot ./bot
RUN pip install -r ./bot/requirements.txt
WORKDIR /usr/src/bot/bot