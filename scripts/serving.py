from flask import Flask, request, jsonify
import numpy as np
from transformers import AutoTokenizer
import torch
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import speculative_sampling

from src import LlamaGPTQ



app = Flask(__name__)
pipeline = None

GLOBAL_SERVER = None


class Server:
    def __init__(self, approx_model_name, target_model_name, tokenizer_model_name) -> None:
        self._device = 'cuda'
        
        logging.info("begin load models")
        self._small_model = LlamaGPTQ.from_quantized(approx_model_name, device="cuda:0")
        self._large_model = LlamaGPTQ.from_quantized(target_model_name, device="cuda:0")
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        logging.info("fininsh load models")
           
        self.num_tokens = 40
        self.top_k = 10
        self.top_p = 0.9
        
    def process_request(self, request : str) -> torch.Tensor:
        input_str = request['prompt']
        logging.info(f"recieve request {input_str}")
        input_ids = self._tokenizer.encode(input_str, return_tensors='pt').to(self._device)
        output = speculative_sampling(input_ids, 
                                      self._small_model, 
                                      self._large_model, self.num_tokens, 
                                      top_k = self.top_k, 
                                      top_p = self.top_p)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Set up a route to listen for inference requests
@app.route('/predict', methods=['POST'])
def predict():
    # Check the content type of the request
    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error': 'Invalid content type'})

    # Get the request data
    request_data = request.json

    # Perform inference
    result = GLOBAL_SERVER.process_request(request_data)

    # Return the inference results
    return jsonify(result)

if __name__ == '__main__':
    GLOBAL_SERVER = Server(
        approx_model_name="/home/chenhj/CodeLlama-7b-4bit",
        target_model_name="/home/chenhj/CodeLlama-7b-4bit",
        tokenizer_model_name="/home/chenhj/CodeLlama-7b-Python-hf"
        )
    # Start the Flask service
    app.run(host='0.0.0.0', port=5000)
