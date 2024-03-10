import logging
import argparse

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='serving arg pars')
    parser.add_argument('--float_model_path', '-f', type=str, default="/raid/chenhj/CodeLlama-7b-Python-hf")
    parser.add_argument('--quant_model_path', '-q', type=str, default="/raid/chenhj/test")

    parser.add_argument('--bits', '-b', type=int, default=4, help='bits num(recommended 4bit)')
    parser.add_argument('--group_size', '-g', type=int, default=128, help='group size(recommended 128)')

    args = parser.parse_args()
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.float_model_path, use_fast=True)
    examples = [
        tokenizer(
            "AI, or Artificial Intelligence, can perform a wide range of tasks, often mimicking human behaviors and capabilities but at a scale or speed that humans cannot match. "
        )
    ]

    quantize_config = BaseQuantizeConfig(
        bits=args.bits, 
        group_size=args.group_size, 
        desc_act=False, 
    )

    model = AutoGPTQForCausalLM.from_pretrained(args.float_model_path, quantize_config)

    model.quantize(examples)
    model.save_quantized(args.quant_model_path)
    model.save_quantized(args.quant_model_path, use_safetensors=True)

    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(args.quant_model_path, device="cuda:0")

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))
