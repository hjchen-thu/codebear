# Codebear
This repository combines **GPTQ 4-bit quantization** and **Speculative Decoding** to accelerate Large Language Models' (LLM) inference in **personal usage scenarios**  (where GPU resources are limited yet there's a pursuit for better performance and faster speed with larger models).

GPTQ is a one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly efficient. And Speculative Decoding is a innovative sampling strategy by using a small approximation model to propose sequences of tokens that will later be checked by a larger model. 

By combining these two techniques, one can even deploy multiple LLMs in a single GPU with limited HBM memory usage. While benefiting from the improved performance brought by larger models, it also helps to accelerate inference speed to some extent.

This repository deploys CodeLlama-34B and CodeLlama-7B in a single V100(32GB) at the same time, with triton-based QuantLinear backend. The total memory usage depends on the number of words already decoded. 

|        | 10 prefill + 200th decoding |
|  ----  | ----  |
| Memory Usage(GB)  | 27.7 | 



<!-- # Update  -->

## Quick Tour

## Future Plans

## References

## Acknowledgement