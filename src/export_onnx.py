import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


LORA_PATH = "./phobert-lora-final"
MERGE_PATH = "./phobert-merged"
ONNX_PATH = "./phobert-merged/model.onnx"
ONNX_DATA_PATH = "./phobert-merged/model.onnx.data"
MAX_LENGTH = 128
NUM_LABELS = 3
DEVICE     = "cpu" 
# ==================================================


def load_lora_model():
    tokenizer  = AutoTokenizer.from_pretrained(LORA_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        LORA_PATH,
        num_labels=NUM_LABELS,
    )
    peft_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    return tokenizer, peft_model


def merge_and_save(peft_model, tokenizer):
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    if hasattr(merged_model, "_hf_peft_config_loaded"):
        merged_model._hf_peft_config_loaded = False

    os.makedirs(MERGE_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGE_PATH)
    tokenizer.save_pretrained(MERGE_PATH)

    return merged_model


def create_dummy_inputs(tokenizer):
    dummy_texts = [
        "Thầy dạy rất hay và nhiệt tình.",
        "Môn học khá nhàm chán và khó hiểu.",
    ]
    dummy_inputs = tokenizer(
        dummy_texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    input_ids = dummy_inputs["input_ids"].to(DEVICE)
    attention_mask = dummy_inputs["attention_mask"].to(DEVICE)
    print(f"\n📐 Dummy inputs shape: {input_ids.shape}")
    return input_ids, attention_mask


def export_onnx(merged_model, input_ids, attention_mask):
    merged_model = merged_model.to(DEVICE)

    torch.onnx.export(
        merged_model,
        args=(input_ids, attention_mask),
        f=ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits":         {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    onnx_size_mb = os.path.getsize(ONNX_PATH) / 1e6
    print(f"✅ ONNX model size: {ONNX_PATH} (~{onnx_size_mb:.1f} MB)")

if __name__ == "__main__":
    tokenizer, peft_model = load_lora_model()
    merged_model          = merge_and_save(peft_model, tokenizer)
    input_ids, attn_mask  = create_dummy_inputs(tokenizer)
    export_onnx(merged_model, input_ids, attn_mask)
    print("\n🎉 Completed!")