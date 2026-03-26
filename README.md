# Vietnamese Student Feedback Classifier

## Project Overview
This project is a classifier for Vietnamese student feedback, utilizing a PhoBERT model served with Triton Inference Server. The project is set up to be run with Docker and Docker Compose.

## Dependencies
The Python dependencies for this project are listed in `exp_requirements.txt`. The main libraries used are:
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `evaluate`
- `torch`
- `scikit-learn`

## Running the project

### 0. Setup Environment
First, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 1. Prepare Dataset (You can pass this step since it was already done)

The dataset needs to be prepared from raw text and label files. Before running this step, download the dataset splits (train/validation/test) from [Hugging Face](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/blob/main/vietnamese_students_feedback.py).

Each split should have 3 files:
- `sents.txt` - Vietnamese sentences
- `sentiments.txt` - Sentiment labels (0=Negative, 1=Neutral, 2=Positive)
- `topics.txt` - Topic labels

Organize them as:
```
data/
├── train/
│   ├── sents.txt
│   ├── sentiments.txt
│   └── topics.txt
├── validation/
│   ├── sents.txt
│   ├── sentiments.txt
│   └── topics.txt
└── test/
    ├── sents.txt
    ├── sentiments.txt
    └── topics.txt
```

Then run the prepare dataset script:
```bash
python src/prepare_dataset.py
```
This will generate CSV files in the `datasets/` folder (train.csv, validation.csv, test.csv).

### 2. Train the Model

You can train the model using either the Jupyter notebook or the training script:

**Option A: Using Jupyter Notebook (Recommended)**
Run all cells to train and fine-tune the PhoBERT model with LoRA (Low-Rank Adaptation). The trained model will be saved to `phobert-lora-final/`.

**Option B: Using the training script directly**
```bash
python src/train.py
```

### 3. Evaluate the Model (Optional)

After training, evaluate the model on the test set:
```bash
python src/evaluate.py
```

### 4. Export Model to ONNX
```bash
python src/export_onnx.py
```

### 5. Run Triton Inference Server

Build and run the server with Docker Compose:
```bash
docker-compose up --build
```
This will:
- Build the Docker image
- Start the Triton Inference Server
- Expose HTTP on port 8000
- Expose gRPC on port 8001
- Expose metrics on port 8002
- Load both `phobert_tokenizer` and `phobert_sentiment` models from `model_repository/`

### 6. Test the Server with Sample Request

Send a test request to the running Triton server:
```bash
python sample_payload.py
```
