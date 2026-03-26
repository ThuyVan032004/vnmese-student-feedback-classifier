import os
from datasets import Dataset


def get_dataset(sentences_txt: str, sentiments_txt: str, topics_txt: str, output_path: str):
    with open(sentences_txt, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f]

    with open(sentiments_txt, "r", encoding="utf-8") as f:
        sentiments = [int(line.strip()) for line in f]

    with open(topics_txt, "r", encoding="utf-8") as f:
        topics = [int(line.strip()) for line in f]

    assert len(sentences) == len(sentiments) == len(topics), (
        f"Length mismatch: sentences={len(sentences)}, "
        f"sentiments={len(sentiments)}, topics={len(topics)}"
    )

    dataset = Dataset.from_dict({
        "sentence": sentences,
        "sentiment": sentiments,
        "topic": topics,
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset.to_csv(output_path)
    print(f"✅ Saved dataset: {output_path} (Num rows: {len(dataset)})")
    return dataset


if __name__ == "__main__":
    # ==================================================
    # User manual
    # ==================================================
    # 1. Download all splits (train/dev/test) from Hugging Face:
    #    https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/blob/main/vietnamese_students_feedback.py
    #
    # 2. With each split (train/dev/test), there are 3 files:
    #    - sents.txt      : sentences
    #    - sentiments.txt : labels for sentiment (0/1/2)
    #    - topics.txt     : labels for topics
    #
    # 3. Put each split's files in the corresponding folder:
    #    - data/train/sents.txt
    #    - data/train/sentiments.txt
    #    - data/train/topics.txt
    # Similarly for validation and test splits.
    # 4. Run this script to generate CSV files in the datasets/ folder:
    #    python prepare_dataset.py
    # ==================================================

    os.makedirs("datasets", exist_ok=True)

    splits = {
        "train": ("data/train/sents.txt", "data/train/sentiments.txt", "data/train/topics.txt", "datasets/train.csv"),
        "validation": ("data/val/sents.txt", "data/val/sentiments.txt", "data/val/topics.txt", "datasets/validation.csv"),
        "test": ("data/test/sents.txt", "data/test/sentiments.txt", "data/test/topics.txt", "datasets/test.csv"),
    }

    for split_name, (sents, sentis, tops, out) in splits.items():
        if os.path.exists(sents):
            print(f"\n[{split_name.upper()}]")
            get_dataset(sents, sentis, tops, out)

    print("\n✅ Completely prepared datasets in the 'datasets/' folder.")