# That one: AI-powered children's story generation

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
```

## Finetuning
Note: train.txt & eval.txt must be in the same folder as finetune.py
```
    python finetune.py --epochs 50 --train_data train.txt --eval_data eval.txt
```