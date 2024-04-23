import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer
from peft import PeftModel, PeftConfig

d = {
    'b': 0,
    't': 1,
    'e': 2,
    'm': 3
}

def load_data(data_path):
    with open(data_path, "r") as f:
        data = f.readlines()
    data = [list(r.strip().split("\t")) for r in data]
    labels = [int(d[r[0]]) for r in data]
    titles = [r[1] for r in data]
    labels = np.array(labels)
    return titles, labels
    
def get_datasetdict(data_path: dict):
    data_dict = {}
    for phase in data_path.keys():
        path = data_path[phase]
        x, y = load_data(path)
        data_dict[phase] = Dataset.from_dict({'text': x, 'labels': y})
    return DatasetDict(data_dict)

# Define which metrics to compute for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
            }

def load_from_checkpoint(path):
    config = PeftConfig.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, return_dict=True, device_map='auto', load_in_8bit = True, num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, path)
    return model, tokenizer

if __name__ == "__main__":
    data_path = "./data/train.txt"
    with open(data_path, "r" ) as f:
        data = f.readlines()    
    print(data[0])