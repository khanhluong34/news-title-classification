import torch 
from torch.utils.data import Dataset

labels = {
    'b': 0,
    'e': 1,
    'm': 2,
    't': 3
}

class NewsDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = f.readlines()
        self.examples = [data[i].strip().split("\t") for i in range(len(data))]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        text = self.examples[idx][0]
        label = labels[self.examples[idx][1]]
        return text, label
    
if __name__ == "__main__":
    dataset = NewsDataset("data/train.txt")
    print(len(dataset))
    print(dataset[0])

        