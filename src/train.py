import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model


MODEL_NAME  = "vinai/phobert-base"
MAX_LENGTH  = 128
NUM_LABELS  = 3
SAVE_PATH   = "./phobert-lora"
SAVE_FINAL_PATH = "./phobert-lora-final"

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

LORA_CONFIG = dict(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],
    bias="none",
)

TRAIN_CONFIG = dict(
    output_dir=SAVE_PATH,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=42,
)
# ==================================================


def load_datasets():
    train_ds = Dataset.from_csv("datasets/train.csv")
    val_ds   = Dataset.from_csv("datasets/validation.csv")
    return train_ds, val_ds


def build_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    total = sum(p.numel() for p in base_model.parameters())
    print(f"Total params in base model: {total:,}")

    lora_config = LoraConfig(**LORA_CONFIG)
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    return tokenizer, peft_model


def tokenize_datasets(tokenizer, train_ds, val_ds):
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_ds = train_ds.rename_column("sentiment", "labels")
    val_ds   = val_ds.rename_column("sentiment", "labels")

    tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["sentence"])
    tokenized_train.set_format("torch")

    tokenized_val = val_ds.map(tokenize, batched=True, remove_columns=["sentence"])
    tokenized_val.set_format("torch")

    return tokenized_train, tokenized_val


def build_compute_metrics():
    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    def compute_metrics(logits_and_labels):
        logits, labels = logits_and_labels
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1  = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        return {"accuracy": acc, "f1_weighted": f1}

    return compute_metrics


def train(tokenizer, peft_model, tokenized_train, tokenized_val):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_args    = TrainingArguments(**TRAIN_CONFIG)

    trainer = Trainer(
        model=peft_model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
    )

    print("\n🚀 Start training...")
    trainer.train()
    print("✅ Training completed!")
    return trainer


def save_model(peft_model, tokenizer):
    import os
    peft_model.save_pretrained(SAVE_FINAL_PATH)
    tokenizer.save_pretrained(SAVE_FINAL_PATH)

    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(SAVE_FINAL_PATH)
        for f in files
    )
    print(f"\n✅ Saved model at: {SAVE_FINAL_PATH}")
    print(f"   Size of LoRA adapter: ~{total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n1. Load dataset...")
    train_ds, val_ds = load_datasets()
    print(f"   Train: {train_ds.num_rows} | Val: {val_ds.num_rows}")

    print("\n2. Initialize model + LoRA...")
    tokenizer, peft_model = build_tokenizer_and_model()

    print("\n3. Tokenize dataset...")
    tokenized_train, tokenized_val = tokenize_datasets(tokenizer, train_ds, val_ds)

    print("\n4. Training...")
    trainer = train(tokenizer, peft_model, tokenized_train, tokenized_val)

    print("\n5. Save model...")
    save_model(peft_model, tokenizer)