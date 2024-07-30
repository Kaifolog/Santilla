import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)
import bentoml
from dotenv import dotenv_values

configs = dotenv_values(".env")


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class Classification:

    def __init__(self) -> None:
        base_model = "sergeyzh/rubert-tiny-turbo"
        model_max_length = 50

        self.tokenizer = BertTokenizer.from_pretrained(
            base_model, model_max_length=model_max_length
        )
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=configs["PATH"],
            local_files_only=True,
        )
        self.model.eval()

    @bentoml.api
    def predict(self, text: str) -> tuple:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()

        return (
            nn.functional.softmax(logits, dim=1).flatten().tolist(),
            self.model.config.id2label[predicted_class_id],
        )
