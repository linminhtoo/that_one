import time
import torch
import numpy as np
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer

CHECKPOINT_FOLDER = "checkpoints/curr_best" # curr_best = nolimit_pat3 # 'gpt2'

def gen_story(input_prompt):
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_FOLDER)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_FOLDER)

    bad_tokens = [np.loadtxt('data/bad_tokens.txt', dtype=int).tolist()]

    device = 0 if torch.cuda.is_available() else -1
    story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)

    print(f'Generating story with prompt {input_prompt}')
    input_prompt = "<BOS> " + input_prompt
    start = time.time()
    story = story_generator(input_prompt, max_length=200, do_sample=True,
                no_repeat_ngram_size=2, repetition_penalty=1.5, temperature=0.85, 
                bad_words_ids=bad_tokens,
                top_p=0.92, top_k=125, early_stopping=True)
    print(f'Time elapsed: {time.time() - start}')
    # better parameters
    # from https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a
    return story[0]['generated_text'][6:] # remove "<BOS> "

if __name__ == '__main__':
    input_prompt = "Once upon a time"
    story = gen_story(input_prompt)
    print(story)

'''
input prompts:
Well, well, well
Knock, knock!
One windy mornng
Why is the window open
Cinderella
The sky is falling
The curtains are blue
Purple apples are happy apples
He buys seashells
Ho! Ho! Ho!
He woke up in cold sweat
Asking a genie for a wish
Once upon a time
Happy New Year!
What shall it be today?
In the beginning (this one quite solid tbh)
Tomorrow will be better
'''

# don't use beam_search, gives very weird results
# original generation settings, results seem inferior/less coherent
# story = story_generator(input_prompt, max_length=200, do_sample=True,
#                repetition_penalty=1.1, temperature=1.2, 
#                top_p=0.95, top_k=50)