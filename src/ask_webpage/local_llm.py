from llama_index.llms import HuggingFaceLLM
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
import torch

def zephyr_7b_alpha(context_window=2048, max_new_tokens=256):
    query_wrapper_prompt = PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
        query_wrapper_prompt=query_wrapper_prompt,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        model_kwargs={"quantization_config": quantization_config, "trust_remote_code": True},
        # tokenizer_kwargs={},
        generate_kwargs={"do_sample": False},
        messages_to_prompt=messages_to_prompt,
        device_map="auto"
    )

    return llm