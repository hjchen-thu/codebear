from flask import Flask, request
from transformers import AutoTokenizer
import torch
import logging
import os
import sys
import json
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import speculative_sampling, LlamaGPTQ

app = Flask(__name__)
GLOBAL_SERVER = None

# reference: https://github.com/feifeibear/LLMSpeculativeSampling
class Server:
    def __init__(self, small_model, large_model, tokenizer_model, max_tokens, top_k, top_p) -> None:
        self._device = "cuda:0"
        self._small_model = LlamaGPTQ.from_quantized(small_model, device="cuda:0", use_triton=True, warmup_triton=False)
        self._large_model = LlamaGPTQ.from_quantized(large_model, device="cuda:0", use_triton=True, warmup_triton=False, inject_fused_attention=False)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
           
        self.num_tokens = max_tokens
        self.top_k = top_k
        self.top_p = top_p
        
    def process_request(self, request : str) -> torch.Tensor:
        input_str = request['prompt']
        logging.info(f"receive request {input_str}")
        input_ids = self._tokenizer.encode(input_str, return_tensors='pt').to(self._device)
        output = speculative_sampling(input_ids, 
                                      self._small_model, 
                                      self._large_model, self.num_tokens, 
                                      top_k = self.top_k, 
                                      top_p = self.top_p)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Set up a route to listen for inference requests
@app.route('/codebear', methods=['POST'])
def codebear():
    # Check the content type of the request
    # if request.headers['Content-Type'] != 'application/json':
    #     return jsonify({'error': 'Invalid content type'})

    # Get the request data
    request_data = request.get_json()
    print(request_data)
    
    # Perform inference
    start_time = time.time()
    result = GLOBAL_SERVER.process_request(request_data)
    end_time = time.time()

    lines = result.splitlines()
    indented_lines = ['    ' + line for line in lines]
    # indented_lines[0] = '    '+indented_lines[0]
    indented_text = '\n'.join(indented_lines)

    max_len = 200
    
    output_data = {
    "response": indented_text,
    "number": max_len,
    "time": "{:.2f}".format(end_time - start_time),
    "tokensps": "{:.2f}".format(max_len / (end_time - start_time)),
    }
    output_json = json.dumps(output_data)
    
    print(indented_text)
        
    return output_json, 200

    # Return the inference results
    # return jsonify(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='serving arg pars')
    parser.add_argument('--small_model', '-s', type=str, default="/raid/chenhj/CodeLlama-7b-4bit")
    parser.add_argument('--large_model', '-l', type=str, default="/raid/chenhj/CodeLlama-34b-4bit")
    parser.add_argument('--tokenizer_model', '-t', type=str, default="/raid/chenhj/CodeLlama-7b-Python-hf")
    parser.add_argument('--max_tokens', '-M', type=int, default=200, help='Max tokens generated')
    parser.add_argument('--top_k', '-k', type=int, default=10, help='top_k')
    parser.add_argument('--top_p', '-p', type=float, default=0.9, help='top_p')
    args = parser.parse_args()
    
    GLOBAL_SERVER = Server(
        small_model=args.small_model,
        large_model=args.large_model,
        tokenizer_model=args.tokenizer_model,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p
        )
    # Start the Flask service
    app.run(host='0.0.0.0', port=5000)
