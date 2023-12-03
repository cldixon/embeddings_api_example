import os 
from typing import List 
from sentence_transformers import SentenceTransformer

#### -------------------------------------------------------------------- ####
#### -- Download specifed models from internet hub(s) _before_ serving -- ####
#### -------------------------------------------------------------------- ####

ARTIFACTS: str = "artifacts"
SPECIFIED_MODELS_FILE: str = "models.txt"



if __name__ == '__main__':

    # -- read specified models from text file
    with open(SPECIFIED_MODELS_FILE, "r") as in_file:
        specified_models = in_file.read().split()
        in_file.close()


    print(f"\nDownloading ({len(specified_models)}) models to `{ARTIFACTS}` directory to make available for API.")

    for checkpoint in specified_models:

        # -- load model from pretrained checkpoint (over internet)
        model = SentenceTransformer(checkpoint)

        # -- create directory if doesn't exist
        checkpoint_dir = os.path.join(ARTIFACTS, checkpoint)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        # -- save model artifacts
        model.save(path=checkpoint_dir)
        print(f"-- `{checkpoint}`")

    print("\nDone. Now the API can run.")