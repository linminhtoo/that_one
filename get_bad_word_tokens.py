''' 
Run this script to get a list of tokens corresponding to a list of common bad words
Important for children!
'''
import itertools
import numpy as np
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
from generate import CHECKPOINT_FOLDER

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_FOLDER) # 'gpt2'

# downloaded from https://www.cs.cmu.edu/~biglou/resources/bad-words.txt
with open('data/bad-words.txt', 'r') as f:
    bad_words = f.readlines()

seen = set(bad_words) # for O(1) find
bad_words_add = ['kill', 'die', 'shit', 'blood', 'arse', 'heck', 'hell', 'idiot', 'stupid']
for word in bad_words_add:
    if word not in seen:
        bad_words.append(word)

bad_tokens =  [tokenizer.encode(word) for word in bad_words]
# print(len(bad_tokens))
# bad_tokens = [[12728], [11979], [16211], [18041], [17208], [258, 694], [12758]]
bad_tokens = np.array(list(itertools.chain.from_iterable(bad_tokens)))
# print(bad_tokens.shape)
np.savetxt('data/bad_tokens.txt', bad_tokens, fmt='%d')

print('Done saving bad_tokens')
# about 5000