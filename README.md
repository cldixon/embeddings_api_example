## Simple Embeddings API Example

This is an example of a barebones **Embeddings API** implementation. The API output mirrors that of the [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings). Built on [FastAPI](https://fastapi.tiangolo.com), [Pydantic](https://docs.pydantic.dev/latest/), and [Sentence Transformers](https://www.sbert.net/index.html), this project is a learning exercise as much as a starting point for developing custom embedding API interfaces.

The API code is purposely contained in a single `main.py` file to keep it flexible, and I have included `TODO` comment tags in places which deserve further development in a production implementation. 

### Get Started

Before running the API, it is recommended to download the model checkpoints which will be used to generate embeddings, so the pretrained weights are not re-downloaded after server restarts, reloads, etc. 

This example API supports serving multiple model options, and these can be specified in the `models.txt` file (a barebones implementation of a model repo). This example comes with (2) specified, pretrained models from the Sentence-Transformers library, but more can be added by simply adding checkpoint names. More checkpoint options are available [here](https://www.sbert.net/docs/pretrained_models.html).

Once the `models.txt` file is ready, run the `download.py` script to save the model weights and configurations in the `artifacts` directory.

### Start the API

The API is built with FastAPI, so start the server as follows:

```sh
$ uvicorn main:app --reload
```

> [!TIP]
> FastAPI provides [Swagger](https://swagger.io/docs) documentation out of the box. These can be reached at `localhost:port/docs#`.

> [!Note]
> Reload allows for editing the source file and updating the app. Not necessary otherwise

### API Endpoints

#### Available Models

This endpoint provides information on the available models (see _specified_ models above), as well as metadata for each model, such as dimension size.

##### Python

```python
import requests
from pprint import pprint 

response = requests.get(url="http://localhost/8000/models", headers={"Content-Type": "application/json"})
pprint(response.json())
```

##### Curl

```sh
$ curl -X 'GET' \
    'http://localhost:8000/models' \
    -H 'accept: application/json'
```

##### Response

```json
{
    "data": {
        "models": {
            "all-MiniLM-L12-v2": {"dim": 384},
            "all-mpnet-base-v2": {"dim": 768},
            ...
        }
    },
    "generated": "2023-12-02 @ 20:39:22",
    "id": "76662a90-55da-47b9-8072-310ed4d090b8"
}
```

#### Text Embeddings

The core feature of the API is to generate and return embedding representations of input text sequences. The example below is for a single text sequence, but the API can handle an array of text sequences as input. The input requires the user to select an available model (see above section). The included `demo.ipynb` notebook contains more usage examples.

The API provides (4) fields of data in the response:

1. `model`: name of the selected model in the request
2. `generated`: generic date/time stamp
3. `id`: generic uuid for record inference
4. `data`: contains the returned embedding contents

For each embedding returned, there are (3) fields of data:

1. `embedding`: vector representation of input text sequence
2. `index`: index number for embedding; relevant if multiple text sequences provided as input
3. `num_tokens`: number of tokens derived from the input text sequence. 

##### Python

```python
import requests

checkpoint = "all-mpnet-base-v2"
text = "Can we mimic the Embeddings API output format from OpenAI? I dunno, but we can try."

response = requests.post(
    url="http://localhost:8000/embeddings",
    json={"model": checkpoint, "input": text},
    headers=headers
)
pprint(response)
```

##### Curl

```sh
$ curl -X 'POST' \
    'http://localhost:8000/embeddings' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "all-mpnet-base-v2",
        "input": "Can we mimic the Embeddings API output format from OpenAI? I dunno, but we can try."
    }'
```

##### Response

```json
{
    "data": [
        {
            "index": 0,
            "embedding": [
                -0.03878581151366234,
                0.03181726858019829,
                -0.020964443683624268,
                ...
            ],
            "num_tokens": 27
        }
    ],
    "generated": "2023-12-02 @ 20:50:04",
    "id": "83e70cf8-e8d6-4783-a928-b45d9507635e",
    "model": "all-mpnet-base-v2"
```


### `TODO's` as Next Steps

Be sure to check out the many `TODO` comments in the `main.py` file to see how this example could be further expanded. Also, submit a pull request to add more `TODO`'s.