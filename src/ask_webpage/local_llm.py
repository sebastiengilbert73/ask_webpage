from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.prompts.promts import SimpleInputPrompt
import torch
from transformers import BitsAndBytesConfig


def messages_to_prompt(messages):  # CF. https://colab.research.google.com/drive/16Ygf2IyGNkb725ZqtRmFQjwWBuzFX_kl?usp=sharing#scrollTo=lMNaHDzPM68f
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt

def zephyr_7b_alpha(context_window=2048, max_new_tokens=256):
    query_wrapper_prompt = PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n")
    model_kwargs = {"trust_remote_code": True}
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs['quantization_config'] = quantization_config

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

def llama2_7b(context_window=4096, max_new_tokens=2048, temperature=0.0):  # Cf. https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoLlama-Local.html
    SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    - Generate human readable output, avoid creating output with gibberish text.
    - Generate only the requested output, don't include any other language before or after the requested output.
    - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    - Generate professional language typically used in business documents in North America.
    - Never generate offensive or foul language.
    """
    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )

    llm = HuggingFaceLLM(
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs={"temperature": temperature, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        model_name="meta-llama/Llama-2-7b-hf",
        device_map="auto",
        # change these settings below depending on your GPU
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
    )
    return llm

def phi2(context_window=4096, max_new_tokens=256, device_map='cuda'):  # Cf. https://gist.github.com/reachrkr/250eaf10b6252b6a936d9abcb67efcca
    system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
    query_wrapper_prompt = SimpleInputPrompt(
        "<|USER|>{query_str}<|ASSISTANT|>"
    )
    llm = HuggingFaceLLM(
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs={"do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="microsoft/phi-2",
        model_name="microsoft/phi-2",
        device_map=device_map,
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    return llm
