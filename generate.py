from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer

CHECKPOINT_FOLDER = "ckpt_stop3"

model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_FOLDER)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_FOLDER)
story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# The format for input_prompt: "<BOS> <genre> Optional text..."
# Supported genres: superhero, sci_fi, horror, thriller, action, drama

input_prompt = "<BOS> It was a sunny morning."
story = story_generator(input_prompt, max_length=150, do_sample=True,
               repetition_penalty=1.1, temperature=1.15, 
               top_p=0.95, top_k=50)
print(story)