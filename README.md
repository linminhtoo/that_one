# TattleTale: AI-powered children's story generation
For iNTUitive hackathon (27-28 Feb 2021) <br>
**Team members (That One): Leow Cong Sheng, Tay Kai Yang, Lin Min Htoo and Soo Jian Xian**

**Slide Deck:**: [https://drive.google.com/file/d/1tt3_8DVBHQjPIkl-QvzQEOXdynA335rZ/view?usp=sharing](https://drive.google.com/file/d/1tt3_8DVBHQjPIkl-QvzQEOXdynA335rZ/view?usp=sharing) <br>
**Video Pitch**: [https://youtu.be/cFYOaKjbBs4](https://youtu.be/cFYOaKjbBs4) <br>
**Dev Post**: [https://devpost.com/software/tattletale](https://devpost.com/software/tattletale) <br>

## Web-app
Check out our webapp here! [https://ta-ttletale.herokuapp.com/](https://ta-ttletale.herokuapp.com/) <br>

<p align="center">
  <img src="tattletale_screenshot.jfif?raw=true" alt="webapp screenshot 1" width=45%>
</p>

## What? <br>
We hope to increase the accessibility and exposure to reading in developing countries.<br>
<br>

## Why? <br>
Because, we strongly believe that education is a social leveller. Reading books and stories have tremendous impacts on cognitive and linguistics development which can be difficult to access for the illiterate.<br>
<br>

## How? <br>
We utilized the GPT-2 language model for story generation through a simple, no-frill web application. To bias the model to generate children's stories, we finetuned the model on a dataset of children's stories that we scraped from the Web. <br>
Frontend: [https://github.com/xfated/intuitive](https://github.com/xfated/intuitive) <br> 
Backend: [https://github.com/linminhtoo/that_one](https://github.com/linminhtoo/that_one) 

## Setup instructions <br>
Just run 
```
    bash -i setup.sh
```
The contents:
``` 
    # ensure conda is already initialized
    conda create -n tattletale python=3.7 -y
    conda activate tattletale

    pip install transformers torch
    pip install beautifulsoup4
```

## Data preparation <br>
Run
```
    python scrape.py
```
Feel free to add more weblinks to scrape. See the code for more details, but the format is: Tuple[link, pointer_for_posts, pointer_for_sentences]

## Finetuning <br>
It takes ~5 minutes to finetune for 3 epochs on 1xRTX2080 using fp16 (mixed precision training). With full precision, it is almost 2x slower. Just 3 epochs of training gave us the best validation loss given our dataset, which is, admittedly, a little small, but works well enough for a proof of concept.
See finetune.py for detailed arguments
```
    python finetune.py \
        --fp16 --epochs 10 --patience 3 \
        --train_data 'data/train_1024_n_80.txt' --eval_data 'data/eval_1024_n_80.txt' \
        --ckpt_folder 'checkpoints/curr_best'
```
To do distributed training over multiple GPUs (replace $N_GPUS with the total number of GPUs per node)
```
    python -m torch.distributed.launch \
        --nproc_per_node $N_GPUS finetune.py \
        --fp16 --epochs 10 --early_stop --patience 3 \
        --train_data 'data/train_1024_n_80.txt' --eval_data 'data/eval_1024_n_80.txt' \
        --ckpt_folder 'checkpoints/curr_best'
```

## Generation <br>
First, download [this entire folder from google drive](https://drive.google.com/drive/folders/1PFBMceE26WG9DeXK_iLu_GnXm6eBYB7A).
Create a folder called ```'checkpoints'``` and drop this folder as-is inside.
You need to specify the checkpoint folder of the weights you wish to use in generate.py, so it should be ```CHECKPOINT_FOLDER = "checkpoints/curr_best"``` in line 4.
See the script itself for parameters that you can modify
```
    python generate.py
```

## Filtering bad words during generation <br>
Especially important since this application is targeted at children. We downloaded a list of ~5000 common inappropriate English words from [CMU CS Website](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt) and appended a few other bad words that we deemed were too violent/rude for children. We have already ran:
```
    python get_bad_word_tokens.py
```
which uses the GPT-2 finetuned tokenizer (GPT-2 tokenizer + ```<BOS>```, ```<EOS>```, ```<PAD>``` tokens). You only need to re-run this script if you
use a different tokenizer, or wish to augment the existing list of bad words. If you change the name of the ```data/bad_tokens.txt``` file, you will need to modify line 12 of ```generate.py```

## Credits <br>
Fine-tuning & generation code were mostly standard [HuggingFace](https://huggingface.co/transformers/) templates. We also referenced the code from [this Towards Data Science article](https://towardsdatascience.com/generate-fresh-movie-stories-for-your-favorite-genre-with-deep-learning-143da14b29d6). All other code, such as scraping & filtering bad words, were written from scratch.

367 children's stories were scraped from two websites:
- [Tonight Bedtime Story](https://www.tonightsbedtimestory.com/stories/)
- [Student UK Bedtime Stories](https://www.studentuk.com/category/bedtime-stories/)

Bad words were downloaded from [Carnegie-Mellon University's Computer Science website](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt)
