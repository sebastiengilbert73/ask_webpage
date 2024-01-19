from llama_index.embeddings import HuggingFaceEmbedding

def bge_small_en_v1p5():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")