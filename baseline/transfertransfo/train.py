import logging
from pprint import pformat
from argparse import ArgumentParser

import torch
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                          GPT2DoubleHeadsModel, GPT2Tokenizer)

# define special tokens
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
# prepare a dict for adding special tokens to tokenizer
# https://huggingface.co/docs/transformers/v4.20.0/en/internal/tokenization_utils#transformers.SpecialTokensMixin.add_special_tokens
ATTR_TO_SPECIAL_TOKENS = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                          'additional_special_tokens': ['<speaker1>', '<speaker2>']}

logger = logging.getLogger(__file__)

def add_special_tokens_(model, tokenizer):
    """Add special tokens to the tokenizer and the model if they have not already been added."""
    # tokenizer.encoder returns a dict, be like: 'hesitancy</w>': 39148
    orig_num_tokens = len(tokenizer.encoder)
    # doesn't add if they are already there
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKENS)
    # make sure to also resize the token embedding matrix of the model so that its embedding
    # matrix matches the tokenizer.
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    # 这里建模的时候可以改变思路，不要直接的history作为input之一去建模，而是将对方给出的人格信息（印象）进行更新保留，自己已透露的人格进行记录
    # 可以先进行一个latent variable的embedding再送入transformer生成？
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    # 这里default不是1e-5有点不理解？
    parser.add_argument("--lr", type=float, default=6.25e-5,help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # logging is set to INFO
    logging.basicConfig(level=logging.INFO)
    # pformat: return the formatted representation of object as a string.
    logger.info("Arguments: %s", pformat(args))

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

if __name__ == "__main__":
    train()