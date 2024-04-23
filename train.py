import os
import random
import functools
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from utils import get_datasetdict, compute_metrics
import os 
from huggingface_hub import login 
import argparse
from dotenv import load_dotenv

# get HuggingFace access token
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
login(token=HUGGINGFACEHUB_API_TOKEN)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.tensor(d['labels']) 
    return d

# Define custom trainer class to be able to calculate multi-class loss
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Set random seed
random.seed(43)

# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

# load model
def get_model(model_name, quantization_config, lora_config, tokenizer, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        num_labels=num_labels
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

# Train

def main(args):
    model_name = args.model_name
    # qunatization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit = args.quantization_8bit, 
        load_in_4bit = args.quantization_4bit,
        bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
        bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
    )

    # lora config
    lora_config = LoraConfig(
        r = args.lora_r, # the dimension of the low-rank matrices
        lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
        # target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout = 0.05, # dropout probability of the LoRA layers
        bias = 'none', # wether to train bias weights, set to 'none' for attention layers
        task_type = 'SEQ_CLS'
    )
    # Load data 
    data_path = args.data_path
    num_labels = args.num_labels
    # Create HF dataset
    ds = get_datasetdict(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')

    model = get_model(model_name, quantization_config, lora_config, tokenizer, num_labels)
    # Define training args
    training_args = TrainingArguments(
        output_dir='./checkpoints',
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.valid_batch_size_per_device,
        logging_dir=args.logging_dir,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        save_strategy='epoch',                           # save the best model based on evaluation metric
        metric_for_best_model='accuracy',               # save the best model with highest accuracy 
        # device='cuda:0',
        report_to=None                                  # if you want to log to wandb, set this to 'wandb'
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['valid'],
        tokenizer=tokenizer,
        data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    # Save model
    multiclass_model_id = f'./checkpoints/{model_name.split("/")[1]}_news_cls'
    trainer.model.save_pretrained(multiclass_model_id)
    tokenizer.save_pretrained(multiclass_model_id)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Train a model for news classification based on the title")
    parse.add_argument("--model-name", type=str, default="openai-community/gpt2", help="The model to use for training")
    parse.add_argument("--train-path", type=str, default="./data/train.txt", help="The path to the training data")
    parse.add_argument("--valid-path", type=str, default="./data/valid.txt", help="The path to the validation data")
    parse.add_argument("--test-path", type=str, default="./data/test.txt", help="The path to the test data")
    parse.add_argument("--train-batch-size-per-device", type=int, default=16, help="The batch size for training")
    parse.add_argument("--valid-batch-size-per-device", type=int, default=128, help="The batch size for validation")
    parse.add_argument("--num_labels", type=int, default=4, help="The number of labels in the dataset")
    parse.add_argument("--lr", type=float, default=1e-4, help="The learning rate for training")
    parse.add_argument("--num-epochs", type=int, default=20, help="The number of epochs to train the model") 
    parse.add_argument("--quantization-mode", type=str, default="8bit", help="The quantization mode to use for training") 
    parse.add_argument("--lora-r", type=int, default=16, help="The dimension of the low-rank matrices")
    parse.add_argument("--logging-dir", type=str, default="./logs", help="The directory to save the logs")
    args = parse.parse_args() 
    # set up following arguments
    args.data_path = {
        "train": args.train_path,
        "valid": args.valid_path,
        "test": args.test_path
    }
    args.quantization_8bit = False if args.quantization_mode == "4bit" else True 
    args.quantization_4bit = False if args.quantization_mode == "8bit" else True 
    main(args)