FROM nvcr.io/nvidia/tritonserver:24.09-py3

RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    protobuf \
    tokenizers