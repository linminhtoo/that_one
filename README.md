# TattleTale: AI-powered children's story generation
For iNTUition hackathon (27-28 Feb 2021) <br>
**Team members (That One): Leow Cong Sheng, Tay Kai Yang, Lin Min Htoo and Soo Jian Xian**

## Setup instructions
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

## Data preparation
Run
```
    python scrape.py
```
Feel free to add more weblinks to scrape. See the code for more details, but the format is: Tuple[link, pointer_for_posts, pointer_for_sentences]

## Finetuning
It takes ~5 minutes to finetune for 3 epochs on 1xRTX2080. This gave us the best validation loss given our dataset, which is, admittedly, a little small.
See finetune.py for detailed arguments
```
    python finetune.py --epochs 30 --patience 3 \
        --train_data 'data/train_1024.txt' --eval_data 'data/eval_1024.txt' \
        --ckpt_folder 'checkpoints/finetune'
```
To do distributed training over multiple GPUs (replace $N_GPUS with the total number of GPUs per node)
```
python -m torch.distributed.launch \
    --nproc_per_node $N_GPUS finetune.py \
    --epochs 30 --early_stop --patience 3 \
    --train_data 'data/train_1024.txt' --eval_data 'data/eval_1024.txt' \
    --ckpt_folder 'checkpoints/finetune'
```

## Generation
First, download [this entire folder from google drive](https://drive.google.com/drive/folders/1PFBMceE26WG9DeXK_iLu_GnXm6eBYB7A).
Create a folder called ```'checkpoints'``` and drop this folder as-is inside.
You need to specify the checkpoint folder of the weights you wish to use in generate.py, so it should be ```CHECKPOINT_FOLDER = "checkpoints/curr_best"``` in line 4.
See the script itself for parameters that you can modify
```
    python generate.py
```

## Filtering bad words during generation
Especially important since this application is targeted at children. We downloaded a list of ~5000 common inappropriate English words from [CMU CS Website](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt) and appended a few other bad words that we deemed were too violent/rude for children. We have already ran:
```
    python get_bad_word_tokens.py
```
which uses the GPT-2 finetuned tokenizer (GPT-2 tokenizer + ```<BOS>```, ```<EOS>``` tokens). You only need to re-run this script if you
use a different tokenizer, or wish to augment the existing list of bad words. If you change the name of the ```data/bad_tokens.txt``` file, you will need to modify line 12 of ```generate.py```

## Credits
Base code was adapted from [this Towards Data Science article](https://towardsdatascience.com/generate-fresh-movie-stories-for-your-favorite-genre-with-deep-learning-143da14b29d6)

367 children's stories were scraped from two websites:
[Tonight Bedtime Story]("https://www.tonightsbedtimestory.com/stories/")
[Student UK Bedtime Stories]("https://www.studentuk.com/category/bedtime-stories/")

Bad words were downloaded from [Carnegie-Mellon University's Computer Science website](https://www.cs.cmu.edu/~biglou/resources/bad-words.txt)