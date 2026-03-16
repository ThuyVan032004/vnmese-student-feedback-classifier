import json
import numpy as np
import requests

payload = {
    "inputs": [{
        "name": "text",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["Thầy cô xinh đẹp, sinh viên rất mê"]
    }]
}

with open("payload.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)

tokenizer_response = requests.post(
    "http://localhost:8000/v2/models/phobert_tokenizer/infer",
    json=payload
)
print("Tokenizer OK:", tokenizer_response.status_code)

tokenizer_output = tokenizer_response.json()["outputs"]
input_ids = tokenizer_output[0]["data"]
attention_mask = tokenizer_output[1]["data"]

sentiment_payload = {
    "inputs": [
        {
            "name": "input_ids",
            "shape": [1, 128],
            "datatype": "INT64",
            "data": input_ids
        },
        {
            "name": "attention_mask",
            "shape": [1, 128],
            "datatype": "INT64",
            "data": attention_mask
        }
    ]
}

sentiment_response = requests.post(
    "http://localhost:8000/v2/models/phobert_sentiment/infer",
    json=sentiment_payload
)
print("Sentiment OK:", sentiment_response.status_code)

logits = sentiment_response.json()["outputs"][0]["data"]
logits = np.array(logits)

labels = ["negative", "neutral", "positive"]

e = np.exp(logits - np.max(logits))
probs = e / e.sum()

for label, prob in zip(labels, probs):
    print(f"{label:10s}: {prob:.1%}")

print(f"\n Prediction result: {labels[np.argmax(logits)]} ({probs.max():.1%})")