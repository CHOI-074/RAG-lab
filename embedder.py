import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def get_model():
  return SentenceTransformer("jhgan/ko-sroberta-multitask")

def build_index(chunks, model):
    embeddings = model.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(query, index, chunks, model, k=3):
    q_emb = model.encode([query]).astype("float32")
    distances, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]