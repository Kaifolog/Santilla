# Santilla

Fine-tuned BERT classifier and fine-tuned T5 containerized with docker compose.

The datasets are composed of chats with my friends (so they are private). The classifier was trained to determine whether a message was written by my particular friend or not. T5 was fine-tuned to mimic this friend's writing style.

## Components

The project consists of three microservices:

1. `сlassifier` is a single endpoint API service implemented with BentoML, utilizing a classifier model based on BERT.
2. `response-gen` is a service implemented with FastAPI and a fine-tuned T5 model for styled response generation.
3. `bot` is a Telegram bot written in Python to provide users with convenient access to the models.

The training processes for the models are described in the `train.ipynb` files.
