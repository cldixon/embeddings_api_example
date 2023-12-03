import os 
from uuid import uuid4 
from datetime import datetime

from fastapi import FastAPI

from typing import Any, List, Tuple 
from typing_extensions import Annotated

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, AfterValidator

import torch 
from sentence_transformers import SentenceTransformer


## -- App Settings ----

## ---- Torch Device Options ------
class DeviceOptions:
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    
def _get_device() -> str:
    """Returns best available device for torch inference."""
    if torch.cuda.is_available():
        return DeviceOptions.CUDA
    elif torch.backends.mps.is_available():
        return DeviceOptions.MPS
    else:
        return DeviceOptions.CPU
    
## ---- Specified Models ------
# ISSUE: this reads from models.txt, but then loads from artifacts/.
# the two could be out of sync. See `download.py`.

ARTIFACTS_DIR = "artifacts"
SPECIFIED_MODELS_FILE = "models.txt"

def load_specified_models(filepath: str = SPECIFIED_MODELS_FILE) -> Tuple[str]:
    """Returns tuple of specified models which API can support for inference."""
    with open(filepath, "r") as in_file:
        specified_models = in_file.read().split()
        in_file.close()
    return tuple(specified_models)


# TODO: model directory/loading can be refactored as an interface (e.g., models stored in blob storage, repo service, etc.)
# TODO: device checking mechanism ties example to pytorch. FYI.
class APISettings(BaseSettings):
    MODEL_DIR: str = Field(default=ARTIFACTS_DIR) # <- we'll want models saved locally before serving. See `download.py`...
    DEVICE: str = Field(default_factory=_get_device)
    SPECIFIED_MODELS: Tuple = Field(default_factory=load_specified_models) # <- these could be your local, custom models


## -- Initialize API + Settings ----
app = FastAPI()
settings = APISettings()


#### -- API Class Models ----

# TODO: application specific uuid generation
def generate_id() -> str:
    """Generate UUID string"""
    return str(uuid4())

# TODO: application specific date/timestamp
def get_timestamp() -> str:
    """Example datetime as string"""
    return datetime.today().strftime("%Y-%m-%d @ %H:%M:%S")


def _check_if_model_is_specified(model: str) -> str:
    """Check if input model is specified as a model to be available.
    Returns error if not, model name if specified."""
    assert model in settings.SPECIFIED_MODELS, f"Model `{model}` has not been specified and is not available."
    return model

AvailableModels = Annotated[str, AfterValidator(_check_if_model_is_specified)]
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

# TODO: currently tied to sentence-transformers library; could re-factor to allow for other libraries, pure pytorch, etc.
def load_model_from_checkpoint(checkpoint: str, device: str | None = None) -> SentenceTransformer:
    """Loads sentence-transformer model from checkpoint. Can be from library's pretrained models
    or local custom models. Also optional provide device if using GPU or Apple Silicon."""
    return SentenceTransformer(checkpoint, device)

# TODO: more sophisticated tokenization options, beyond truncation; support chunking, windows sizes, etc.
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

# TODO: a dedicated class for handling multiple models; strategies for models from different libraries, etc.
# TODO: use caching to effectively manage model in memory, when called, etc.
model_map = {
    checkpoint: load_model_from_checkpoint(
        checkpoint=os.path.join(settings.MODEL_DIR, checkpoint),
        device=settings.DEVICE
    )
    for checkpoint in settings.SPECIFIED_MODELS
}

# TODO: this could be re-factored into dedicated class.
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
# TODO: would need to account for auth, limits, etc.
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