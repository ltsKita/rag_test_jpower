from model_download import get_tokenizer_elyza_8b, get_model_elyza_8b
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import re

tokenizer_elyza = get_tokenizer_elyza_8b()
model_elyza = get_model_elyza_8b()

def get_llm():
    pipe = pipeline(
        "text-generation",
        model=model_elyza,
        tokenizer=tokenizer_elyza,
        device=0,
        # no_repeat_ngram_size=2
        )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# print(llm("富士山の高さを教えてください"))