import os 
import torch 
from uuid import uuid4 
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel, Field 
from pydantic_settings import BaseSettings
from typing import Any, List, Literal, Optional, Tuple 
from sentence_transformers import SentenceTransformer
# test
## -- App Settings ----
class APISettings(BaseSettings):
    MODEL_DIR: str = "artifacts" # <- we'll want models saved locally before serving. See `download.py`...
    DEVICE: Literal["cpu", "device", "mps"] = "cpu"
    AVAILABLE_CHECKPOINTS: Tuple = ("all-mpnet-base-v2", "all-MiniLM-L12-v2") # <- these could be your local, custom models


## -- Initialize API + Settings ----
app = FastAPI()
settings = APISettings()


#### -- API Class Models ----
def generate_id() -> str:
    """Generate UUID string"""
    return str(uuid4())

def get_timestamp() -> str:
    """Example datetime as string"""
    return datetime.today().strftime("%Y-%m-%d @ %H:%M:%S")


AvailableModels = Literal["all-mpnet-base-v2", "all-MiniLM-L12-v2"]
Generated: str = Field(default_factory=get_timestamp)
UUID: str = Field(default_factory=generate_id)

## ---- User Input Class ------
class UserInput(BaseModel):
    model: AvailableModels
    input: str | List[str]

class Embedding(BaseModel):
    index: int 
    embedding: List[float]
    num_tokens: int 

class ModelOptions(BaseModel):
    models: dict 

class EmbeddingResult(BaseModel):
    model: AvailableModels
    embeddings: List[Embedding] = Field(..., repr=False)


## ---- API Output Class ------
class APIOutput(BaseModel):
    data: Any 
    model: AvailableModels | None = Field(default=None)
    id: str = UUID 
    generated: str = Generated
    

## -- Embedding Model Functions ----

def load_model_from_checkpoint(checkpoint: str, device: str | None = None) -> SentenceTransformer:
    """Loads sentence-transformer model from checkpoint. Can be from library's pretrained models
    or local custom models. Also optional provide device if using GPU or Apple Silicon."""
    return SentenceTransformer(checkpoint, device)

def get_embeddings(model: SentenceTransformer, text: str | List[str]) -> List[Embedding]:
    """Returns sentence embedding objects fro input text. Output will be a list even
    if input is a single (i.e., unwrapped) text sequence."""

    # -- tokenization and inference ----
    wrapped_text = ([text] if isinstance(text, str) else text)
    ## NOTE: perform tokenization + inference separately (rather than `model.encode()`) so we can utilize `input_ids` later...
    tokenized = model.tokenize(wrapped_text)
    with torch.no_grad():
        outputs = model(tokenized)

    # -- unpack + type conversion, shape check ----
    embeddings = outputs["sentence_embedding"].tolist()
    input_ids = tokenized["input_ids"].tolist()
    assert len(embeddings) == len(input_ids), f"Shape mismatch. Size of 'embeddings' (len={len(embeddings)}) should equal 'input_ids' (len={len(input)})."

    # -- format for API output ----
    return [
        Embedding(
            index=i,
            embedding=emb,
            num_tokens=len([
                _id for _id in tokens if _id != model.tokenizer.pad_token_id # remove padding which inflates token count
            ])
        )
        for i, (emb, tokens) in enumerate(zip(embeddings, input_ids))
    ]


## -- Initialize Models and Metadata ----
model_map = {
    checkpoint: load_model_from_checkpoint(os.path.join(settings.MODEL_DIR, checkpoint))
    for checkpoint in settings.AVAILABLE_CHECKPOINTS
}

# TODO: this could be re-implemented as something more powerful than a dictionary.
# TODO: additional metadata could include last trained date, metrics, etc.
model_metadata = {
    checkpoint: {"dim": model.get_sentence_embedding_dimension()}
    for checkpoint, model, in model_map.items()
}


## -- API Endpoints ----

@app.get("/")
def read_root():
    return {"greeting": "Hello, you've reached a very generic text embeddings API!"}

## ---- get available models ------
@app.get("/models", response_model_exclude_none=True)
def available_models() -> APIOutput:
    return APIOutput(
        data=ModelOptions(models=model_metadata)
    )

## ---- get text embeddings ------
@app.post("/embeddings", response_model_exclude_none=True)
def model_inference(user_input: UserInput) -> APIOutput:
    user_input = user_input.model_dump()

    embeddings = get_embeddings(
        model=model_map.get(user_input.get("model")),
        text=user_input.get("input")
    )

    return APIOutput(
        data=embeddings,
        model=user_input.get("model")
    )
