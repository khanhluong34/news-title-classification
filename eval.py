import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer
from peft import PeftModel, PeftConfig

from utils import get_datasetdict, compute_metrics, load_from_checkpoint
import functools
from train import collate_fn, tokenize_examples, CustomTrainer
import argparse

def eval(args):
    checkpoint_path = args.checkpoint_path
    eval_data_path = args.eval_data_path
    
    # Load the model and tokenizer
    model, tokenizer = load_from_checkpoint(checkpoint_path)
    
    # Get eval dataset
    ds = get_datasetdict(eval_data_path)
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')
    
    # Load the Lora model
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    eval_metrics = trainer.evaluate(tokenized_ds['test'])
    # save the eval_results
    with open(args.output_dir, "w") as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")
    print(eval_metrics)
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser("Run a model evaluation on the test set")
    parse.add_argument("--checkpoint-path", type=str, default="checkpoints/gpt2_news_cls",help="Path to the model checkpoint")
    parse.add_argument("--eval-data-path", type=str, default="./data/test.txt", help="Path to the evaluation data")
    parse.add_argument("--output-dir", type=str, default="eval_result.txt", help="Path to save the evaluation results")
    args = parse.parse_args()
    args.eval_data_path = {
        'test': args.eval_data_path
    }
    eval(args)