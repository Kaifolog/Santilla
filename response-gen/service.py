import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import dotenv_values

configs = dotenv_values(".env")

from fastapi import FastAPI

app = FastAPI()

model_name = "cointegrated/rut5-base-multitask"
model_max_length = 64

tokenizer = T5Tokenizer.from_pretrained(
    model_name,
    model_max_length=model_max_length,
)
model = T5ForConditionalGeneration.from_pretrained(
    configs["PATH"],
    local_files_only=True,
)
model.generation_config.max_new_tokens = 72


def generate_output(text):
    inputs = tokenizer(f"answer |{text}", return_tensors="pt")
    with torch.no_grad():
        hypotheses = model.generate(
            **inputs,
            num_beams=10,
            do_sample=True,
            top_p=0.7,
            num_return_sequences=3,
            repetition_penalty=2.5,
        )
    return tokenizer.batch_decode(hypotheses, skip_special_tokens=True)


@app.post("/generate")
def generate(text: str) -> list[str]:
    return generate_output(text)
