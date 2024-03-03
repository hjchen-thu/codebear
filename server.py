import torch
import os
import json
import time
import transformers
from transformers import AutoTokenizer
from flask import Flask, request

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

app = Flask(__name__)

model = "/home/chenhj/CodeLlama-7b-Python-hf"
# model = "/home/chenhj/CodeLlama-13b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


@app.route('/codebear', methods=['POST'])
def codebear():
    # data = request.get_json()
    data = request.get_json()
    # modified_binary_data = binary_data.replace(b'\n', b'\\n')
    # print(binary_data)
    # print(modified_binary_data)
    # data = json.loads(binary_data.decode('utf-8')[:-2])
    # data = json.loads(binary_data.decode('utf-8').strip())

    # return a
    try:
        prompt = data.get("prompt", "")
        
        # max_len = data['max_len']
        # temperature = data['temperature']
        # top_k = data['top_k']
        # top_p = data['top_p']
        print(prompt)
        max_len = 200
        
        start_time = time.time()
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            temperature=0.1,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_len,
        )
        end_time = time.time()
        

        
        original_text = sequences[0]['generated_text']
        lines = original_text.splitlines()
        indented_lines = ['    ' + line for line in lines]
        # indented_lines[0] = '    '+indented_lines[0]
        indented_text = '\n'.join(indented_lines)

        output_data = {
        "response": indented_text,
        "number": max_len,
        "time": "{:.2f}".format(end_time - start_time),
        "tokensps": "{:.2f}".format(max_len / (end_time - start_time)),
        }
        output_json = json.dumps(output_data)
        
        print(indented_text)
        
        return output_json, 200
    except KeyError:
        return 'Bad Request, JSON object not valid', 400
    except TypeError:
        return 'Request data is not JSON', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)