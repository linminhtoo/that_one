from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer

""" 
Below, my model checkpoint is commented out. You can replace your checkpoint 
with that to test story generation if your checkpoint didn't train for long enough
"""
#checkpoint = "pranavpsv/gpt2-genre-story-generator"
checkpoint = "checkpoint"

model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# The format for input_prompt: "<BOS> <genre> Optional text..."
# Supported genres: superhero, sci_fi, horror, thriller, action, drama

input_prompt = "<BOS> It was a sunny morning."
story = story_generator(input_prompt, max_length=150, do_sample=True,
               repetition_penalty=1.1, temperature=1.15, 
               top_p=0.95, top_k=50)
print(story)