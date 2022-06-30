from datetime import datetime
import logging
import os
import json
import socket

import torch

from transformers import cached_path

# TODO: use original txt file to convert to the json file format with the features we like
PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

logger = logging.getLogger(__file__)


def get_dataset(tokenizer, dataset_path, dataset_cache):
    """Get tokenized PERSONACHAT dataset from S3 or cache."""
    # if it is dataset_path, then the dataset is not tokenized
    dataset_path = dataset_path or PERSONACHAT_URL
    # To avoid using GPT cache for GPT-2 and vice versa
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        # Loads an object saved with torch.save() from a file.
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


# After Python 3.5, you can assign the parameter's type, if 'model_name' is not a str, you will get a warning.
def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # NOTE: socket?
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir