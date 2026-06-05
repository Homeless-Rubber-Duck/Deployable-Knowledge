from __future__ import annotations
from functools import lru_cache
import ollama
from typing import Any, Dict, List, Optional

# Set your Ollama-compatible model ID 
# (e.g., 'bge-large', 'mxbai-embed-large', or 'all-minilm')
EMBEDDING_MODEL_ID = "mxbai-embed-large"

@lru_cache(maxsize=1)
def check_and_pull_model(model_id: str = EMBEDDING_MODEL_ID):
    """Ensures Ollama has downloaded the model locally."""
    try:
        # Check if the server is alive and pulling is required
        ollama.show(model_id)
    except ollama.ResponseError:
        print(f"Model '{model_id}' not found locally. Fetching via Ollama...")
        ollama.pull(model_id)

def get_text_embeddings(sentences: list[str | Any], model_id: str = EMBEDDING_MODEL_ID) -> list[list[float]]:
    """Generates a batch of embeddings by calling the local Ollama API.
    
    Accepts a list of strings and returns a list of vector arrays.
    """
    # Quick type-safety check if 'Any' objects are passed in the list
    clean_sentences = [str(s) for s in sentences]
    
    # Ensure the model is downloaded and ready
    check_and_pull_model(model_id)
    
    response = ollama.embed(
        model=model_id,
        input=clean_sentences  # Ollama accepts a list of strings here
    )
    
    # Ollama returns a list of vectors inside the 'embeddings' key
    # e.g., [[0.1, 0.2, ...], [0.5, -0.1, ...]]
    return response['embeddings']
    #Old functions load_embedding_model
    #fetch_model_if_needed