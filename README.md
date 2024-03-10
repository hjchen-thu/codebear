# Codebear
This repository combines **GPTQ 4-bit quantization** and **Speculative Decoding** to accelerate Large Language Models' (LLM) inference in **personal usage scenarios**  (where GPU resources are limited yet there's a pursuit for better performance and faster speed with larger models).

GPTQ is a one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly efficient. And Speculative Decoding is a innovative sampling strategy by using a small approximation model to propose sequences of tokens that will later be checked by a larger model. 

By combining these two techniques, one can even deploy multiple LLMs in a single GPU with limited HBM memory usage. While benefiting from the improved performance brought by larger models, it also helps to accelerate inference speed to some extent.

The flowing figures are tested in a single V100(32GB) by deploying CodeLlama-34B and CodeLlama-7B models, with triton-based QuantLinear backend.

|        | 10 prefill + 200th decoding |
|  ----  | ----  |
| Memory Usage(GB)  | 27.7 | 


| 10 prefill + 200th decoding| CodeLlama 7B(FP16) |CodeLlama 7B(4Bit) |CodeLlama 34B(4Bit) |Speculative 7B+34B(4Bit)|
|  ----  | ----  |----  |----  |----  |
| Inference Speed(Tokens/sec)  | 14.3 | 34.1 | 7.9 | 10.2 | 



<!-- # Update  -->

## Quick Tour

## Future Plans

## References

## Acknowledgement