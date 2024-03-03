import torch
import os
import json
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request

# os.environ["CUDA_VISIBLE_DEVICES"] = "15"


prompt = ' socket\n\ndef ping_exponential_backoff(host: str):'
checkpoint = "/home/chenhj/CodeLlama-7b-Python-hf"
# checkpoint = "/home/chenhj/CodeLlama-13b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# inputs = tokenizer(prompt, return_tensors="pt")
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to('cuda')
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=path,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
a = 1    
