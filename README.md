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
1. **Run the notebook to get fine-tuned model**

2. **Build and run the Triton server:**
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker image and start the Triton Inference Server. The server will expose HTTP on port 8000, gRPC on port 8001, and metrics on port 8002. The models are located in the `model_repository` directory.

## Sample Request
You can send a request to the server using the `sample_payload.py` script. This script first tokenizes the text using the `phobert_tokenizer` model and then sends the tokenized input to the `phobert_sentiment` model for classification.

To run the sample request:
```bash
python sample_payload.py
```
This will send a sample request to the Triton server and print the response.
