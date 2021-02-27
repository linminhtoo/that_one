import logging
import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import EarlyStoppingCallback

# Setup logging
logger = logging.getLogger(__name__)

# Get access to model types and model configs to select GPT2 model and config
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser("finetune.py")
    # mode & metadata
    parser.add_argument("--expt_name", help="experiment name", type=str, default=datetime.strftime(datetime.now(), "%y%m%d-%H%Mh"))
    parser.add_argument("--local_rank", help="local rank for distributed training", type=int, default=-1)
    # file names
    parser.add_argument("--log_file", help="log_file", type=str, default="train")
    parser.add_argument("--train_data", help="train_data", type=str, default="data/train_1024_n.txt")
    parser.add_argument("--eval_data", help="eval_data", type=str, default="data/eval_1024_n.txt")
    parser.add_argument("--ckpt_folder", help="checkpoint_folder", type=str, default="checkpoints/ckpt")
    # training params
    parser.add_argument("--bs", help="batch size", type=int, default=4) # 8 will OOM on 1xRTX2080
    parser.add_argument("--patience", help="patience for early stopping", type=int, default=5)
    # parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="num. of epochs", type=int, default=50)
    parser.add_argument("--fp16", help="use fp16", action="store_true")
    # model params
    # nothing for now

    return parser.parse_args()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "Train with masked-language modeling loss instead of language modeling."
        },
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

# Create LineByLineDataset from Movie Plots text file
def get_dataset(
    args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
        )

def main(args):
    model_args = ModelArguments(
        model_name_or_path="gpt2", model_type="gpt2"
    )
    data_args = DataTrainingArguments(
        train_data_file=args.train_data, #"scrapped.txt",
        eval_data_file=args.eval_data, #"scrapped.txt",
        line_by_line=True,
        block_size=512,
        overwrite_cache=True,
    )
    training_args = TrainingArguments(
        output_dir=args.ckpt_folder, # "checkpoint",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        logging_steps=200,
        logging_dir=f'./logs/{args.expt_name}', # directory for storing logs
        per_device_train_batch_size=args.bs,
        num_train_epochs=args.epochs,
        save_total_limit=1,
        # save_steps=1000, # ignored by load_best_model_at_end
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        fp16=args.fp16,
        local_rank=args.local_rank
    )

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed for deterministic training runs
    set_seed(training_args.seed)


    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        # "additional_special_tokens": [
        #     "<story>"
        # ],
    }

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0: 
      # If block_size <= 0, set it to max. possible value allowed by model
        data_args.block_size = tokenizer.model_max_length # NOTE: MinHtoo: had to change from tokenizer.max_len
        # see https://github.com/huggingface/transformers/issues/8739
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        # prediction_loss_only=True,
    )

    # Training
    try:
      if training_args.do_train:
          model_path = (
              model_args.model_name_or_path
              if model_args.model_name_or_path is not None
              and os.path.isdir(model_args.model_name_or_path)
              else None
          )
          trainer.train(model_path=model_path)
          trainer.save_model()
          tokenizer.save_pretrained(training_args.output_dir)
    except KeyboardInterrupt:
      print("Saving model that was in the middle of training")
      trainer.save_model()
      tokenizer.save_pretrained(training_args.output_dir)
      return

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    cmd_args = parse_args()
    # logger.info(f'{cmd_args}')
    os.makedirs(f'./logs/{cmd_args.expt_name}', exist_ok=True)
    main(cmd_args)
