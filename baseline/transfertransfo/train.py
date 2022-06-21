"""
code is from How to build a State-of-the-Art
Conversational AI with Transfer learning
(https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)
"""
from transformers import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from itertools import chain

# Let's define our contexts and special tokens
persona = [["i", "like", "playing", "football", "."],
           ["i", "am", "from", "NYC", "."]]
history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]
reply = ["great", "to", "hear"]
bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"


def build_inputs(persona, history, reply):
    # Build our sequence by adding delimiters and concatenating
    # use list(chain(*persona)) to iterate every word in persona.
    # sequence[0]: persona; sequence[1]: history[0]; sequence[2]: history[1];
    # sequence[2]: reply[0]
    sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
    # add <speaker1> and <speaker2>
    sequence = [sequenc[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] +
                               s for i, s in enumerate(sequence[1:])]
    # Build our word, segments and position inputs from sequence
    # word tokens
    words = list(chain(*sequence))
    # segment tokens
    segments = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    # position tokens(absolute)
    position = list(range(len(words)))
    return words, segments, position, sequence


words, segments, position, sequence = build_inputs(persona, history, reply)

# >>>print(sequence) # Our inputs looks like this:
# [['<bos>', 'i', 'like', 'playing', 'football', '.', 'i', 'am', 'from', 'NYC', '.'],
# ['<speaker1>', 'hello', 'how', 'are', 'you', '?'],
# ['<speaker2>', 'i', 'am', 'fine', 'thanks', '.'],
# ['<speaker1>', 'great', 'to', 'hear', '<eos>']]

# 'openai-gpt' is checkpoint
model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

# add special tokens to vocabulary
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
tokenizer.set_special_tokens(SPECIAL_TOKENS)
model.set_num_special_tokens(len(SPECIAL_TOKENS))

# Tokenize words and segments embeddings:
words = tokenizer.convert_tokens_to_ids(words)
segments = tokenizer.convert_tokens_to_ids(segments)

