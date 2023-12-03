import json 
import requests 
import argparse
from pprint import pprint

## set up arguments
parser = argparse.ArgumentParser()

## -- url
parser.add_argument(
    "--url", 
    type=str, 
    default="http://localhost:8000", 
    help="Hostname and port number for Fastapi server."
)

## -- models endpoint
parser.add_argument(
    "--models",
    type=str,
    default="models",
    help="Name of endpoint to check available models."
)

## embeddings endpoint
parser.add_argument(
    "--embeddings",
    type=str,
    default="embeddings",
    help="Name of endpoint for generating text embeddings."
)

## finalize arguments
args = parser.parse_args()


HEADERS = {
    "Content-Type": "application/json"
}


if __name__ == '__main__':

    print("---- [EMBEDDINGS API TESTING] ----")
    print(f"url: `{vars(args).get('url')}`\n")

    ## -- available models endpoint ----
    print(f"-- [AVAILABLE MODELS ENDPOINT] --")
    url = f"{args.url}/{args.models}"
    print(f"url: {url}")

    response = requests.get(url=url, headers=HEADERS)
    print(response)

    if response.status_code == 200:
        content = response.json()
        pprint(content)
        available_models = [model for model in content["data"]["models"]]
        print(f"({len(available_models)}) available models; will test all.")

    else:
        print(response.content)
    print('-' * 50, '\n')

    ## -- single text input ----
    print(f"-- [SINGLE TEXT SEQUENCE] --")

    embeddings_url = f"{args.url}/{args.embeddings}"
    print(f"url (for embeddings): {embeddings_url}\n")

    single_text = "Can we mimic OpenAI's embeddings API ouputs? I dunno, let's see."

    for model in available_models:
        print(f"---> testing model `{model}`")
        response = requests.post(
            url=embeddings_url,
            json={"model": model, "input": single_text},
            headers=HEADERS
        )
        print(response)
        print(response.json().keys())
        print('-' * 50, '\n')

    ## -- multiple text input ----
    print("-- [MULTIPLE TEXT SEQUENCES] --\n")
    multi_text = [single_text, "well, how does it look?"]

    for model in available_models:
        print(f"---> testing model `{model}`")
        response = requests.post(
            url=embeddings_url,
            json={
                "model": model,
                "input": multi_text
            },
            headers=HEADERS
        )

        print(response)
        if response.status_code == 200:
            content = response.json()
            num_embeddings = len(content['data'])
            assert num_embeddings == len(multi_text), f"Returned {num_embeddings} embeddings for {len(multi_text)} text examples."
            print(f"** returned {len(content['data'])} embeddings...")
    
    print('-' * 50, '\n')

    print("-- [INVALID MODEL SELECTION] --")
    model = "i-ll-take-the-finest-model-you-ve-got!"
    text = "..."

    print(f"expecting api error for model selection `{model}`.")

    response = requests.post(
        url=embeddings_url,
        json={"model": model, "input": text},
        headers=HEADERS
    )

    if response.status_code == 422:
        error_msg = json.loads(response.content)
        pprint(error_msg)
    else:
        print(f"ERROR: Expected HTTP error 422, but instead received HTTP status {response.status_code}.")


    print("\n[DONE].")