# That one: AI-powered children's story generation
For iNTUition hackathon (27-28 Feb 2021)
Team members: Leow Cong Sheng, Tay Kai Yang, Lin Min Htoo and Soo Jian Xian

## Setup instructions
Just run 
```
    bash -i setup.sh
```
The contents:
``` 
    # ensure conda is already initialized
    conda create -n that_one python=3.7 -y
    conda activate that_one

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
Finetuning is fast, takes <2 min for 50 epochs on 367 stories on 1x RXT2080
```
    python finetune.py --epochs 50 --train_data 'data/train.txt' --eval_data 'data/eval.txt' --ckpt_folder 'checkpoint'
```

## Generation
You need to specify the checkpoint folder of the weights you wish to use in generate.py.
See the script itself for parameters that you can modify
```
    python generate.py
```

## Credits
Adapted from [this awesome article](https://towardsdatascience.com/generate-fresh-movie-stories-for-your-favorite-genre-with-deep-learning-143da14b29d6)