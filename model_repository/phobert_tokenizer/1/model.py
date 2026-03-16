import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils

MODEL_PATH = "/models/phobert_tokenizer/1/tokenizer_files"
MAX_LENGTH = 128

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            decoded_text = [text.decode("utf-8") for text in text_tensor.as_numpy().flatten()] # .flatten() makes sure the code works right even if the input is a batch of strings
            
            enc = self.tokenizer(
                decoded_text,
                return_tensors="np",
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
            )
            
            input_ids = pb_utils.Tensor("input_ids", enc["input_ids"].astype(np.int64))
            
            attention_mask_tensor = pb_utils.Tensor(
                "attention_mask", enc["attention_mask"].astype(np.int64)
            )
            
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[input_ids, attention_mask_tensor]
                )
            )
            
        return responses
    
    def finalize(self):
        pass