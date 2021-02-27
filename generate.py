import time
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer

CHECKPOINT_FOLDER = "checkpoints/curr_best" # curr_best = 1024_4gpu_multi_stop5 # "ckpt_bs16_gpu4" # "ckpt_testing" # "ckpt_nolinebyline_stop3"

model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_FOLDER)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_FOLDER)
story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# The format for input_prompt: "<BOS> Optional text..."

input_prompt = "<BOS> Knock knock. "
start = time.time()
print(f'Generating story with prompt {input_prompt}')
story = story_generator(input_prompt, max_length=200, do_sample=True,
               repetition_penalty=1.1, temperature=1.2, 
               top_p=0.95, top_k=50)

print(story)
print(f'Time elapsed: {time.time() - start}')

'''
input prompts:
Knock knock.
