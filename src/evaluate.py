import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel


FINAL_PATH = "./phobert-lora-final"
MAX_LENGTH = 128
NUM_LABELS = 3
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def load_test_dataset():
    return Dataset.from_csv("datasets/test.csv")


def load_model_and_tokenizer():
    tokenizer  = AutoTokenizer.from_pretrained(FINAL_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        FINAL_PATH,
        num_labels=NUM_LABELS,
    )
    peft_model = PeftModel.from_pretrained(base_model, FINAL_PATH)
    peft_model.eval()
    return tokenizer, peft_model


def prepare_test_data(tokenizer, test_ds):
    test_ds = test_ds.rename_column("sentiment", "labels")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    tokenized = test_ds.map(tokenize, batched=True, remove_columns=["sentence"])
    tokenized.set_format("torch")
    return tokenized


def run_prediction(tokenizer, peft_model, tokenized_test):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=64,
        report_to="none",
    )
    trainer = Trainer(
        model=peft_model,
        args=train_args,
        data_collator=data_collator,
    )

    outputs   = trainer.predict(tokenized_test)
    preds     = np.argmax(outputs.predictions, axis=-1)
    true_labels = outputs.label_ids
    return preds, true_labels


def plot_confusion_matrix(true_labels, preds):
    cm = confusion_matrix(true_labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Prediction Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: confusion_matrix.png")


def print_report(true_labels, preds):
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=LABEL_NAMES))


if __name__ == "__main__":
    print("1. Load test dataset...")
    test_ds = load_test_dataset()
    print(f"   Test: {test_ds.num_rows} mẫu")

    print("\n2. Load model + tokenizer from", FINAL_PATH)
    tokenizer, peft_model = load_model_and_tokenizer()

    print("\n3. Tokenize test data...")
    tokenized_test = prepare_test_data(tokenizer, test_ds)

    print("\n4. Run prediction...")
    preds, true_labels = run_prediction(tokenizer, peft_model, tokenized_test)

    print("\n5. Results:")
    print_report(true_labels, preds)
    plot_confusion_matrix(true_labels, preds)