from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# モデルとトークナイザをダウンロードしてローカルのmodelディレクトリに保存
model_name_swallow_70b = "tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1"
model_name_elyza_8b = "elyza/Llama-3-ELYZA-JP-8B"
model_name_embedding_model = "intfloat/multilingual-e5-large"
# モデル格納先ディレクトリを指定
model_dir = "/datadrive/model" 

# トークナイザーとモデルを指定ディレクトリにキャッシュ
# tokenizer_swallow_70b = AutoTokenizer.from_pretrained(model_name_swallow_70b, cache_dir=model_dir)
# model_swallow_70b = AutoModelForCausalLM.from_pretrained(model_name_swallow_70b, cache_dir=model_dir)
tokenizer_elyza_8b = AutoTokenizer.from_pretrained(model_name_elyza_8b, cache_dir=model_dir)
model_elyza_8b = AutoModelForCausalLM.from_pretrained(model_name_elyza_8b, cache_dir=model_dir)

# Embeddingsのモデルを事前にキャッシュ
embedding_model = SentenceTransformer(model_name_embedding_model, cache_folder=model_dir)

# Embeddingsのセットアップ
embeddings = HuggingFaceEmbeddings(model_name=model_name_embedding_model)


# def get_tokenizer_swallow_70b():
#     return tokenizer_swallow_70b

# def get_model_swallow_70b():
#     return model_swallow_70b

def get_tokenizer_elyza_8b():
    return tokenizer_elyza_8b

def get_model_elyza_8b():
    return model_elyza_8b

def get_embeddings():
    return embeddings