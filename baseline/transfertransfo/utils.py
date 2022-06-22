PERSONA_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

def get_dataset(tokenizer, dataset_path, dataset_cache):
    """Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache =