# News Classification Model Training

This repository features a simple FastAPI deployment endpoint for news classification and a PyTorch setup for training and evaluation using Hugging Face Transformers. We've employed a custom dataset and advanced techniques like quantization LoRA for cost-saving in limited-resource computing.

## Setup

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)
- CUDA-enabled GPU (optional for faster training)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/khanhluong34/news-title-classification.git
    cd news-title-classification
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Dataset

The dataset comprises three text files: `train.txt`, `valid.txt`, and `test.txt`, each containing news titles and their respective labels. Please ensure the dataset is stored within the `data` directory. To obtain the text files, run the notebook `process_data.ipynb` located in the `data` directory. Additionally, the notebook provides statistical analysis of the data, including metrics such as the average number of words in samples.

### Model Selection

You can specify the model architecture for training using the --model-name argument. By default, the model is 'openai-community/gpt2' for cost and time-saving.

### Training

To train the model, run the following command:

```bash
python train.py --model-name <model_name> --train-path <train_path> --valid-path <valid_path> --test-path <test_path> --train-batch-size-per-device <train_batch_size> --valid-batch-size-per-device <valid_batch_size> --num_labels <num_labels> --lr <learning_rate> --num-epochs <num_epochs> --quantization-mode <quantization_mode> --lora-r <lora_r> --logging-dir <logging_dir>
```

Replace the placeholders with appropriate values. For example:

```bash
python train.py --model-name openai-community/gpt2 --train-path ./data/train.txt --valid-path ./data/valid.txt --test-path ./data/test.txt --train-batch-size-per-device 16 --valid-batch-size-per-device 128 --num_labels 4 --lr 1e-4 --num-epochs 20 --quantization-mode 8bit --lora-r 16 --logging-dir ./logs
```

### Evaluation

After training, the model checkpoints are saved in the `checkpoints` directory. You can evaluate the model using the test dataset by running:

```bash
python eval.py --checkpoint-path <checkpoint_path> --test-path <test_path>
```

Replace `<checkpoint_path>` with the path to the saved model checkpoint and `<test_path>` with the path to the test dataset.

### Deploy endpoint

To deploy the endpoint for news classification using FastAPI, follow these steps:
Open the `app.py` file and ensure that the model loading code (load_from_checkpoint) correctly points to the location of your trained model checkpoint (define the `checkpoint_path`) then run:
```
python app.py
```
Once the server is running, you can access the API endpoint at http://localhost:2005. 
For friendly UI testing, you can access http://localhost:2005/docs 
Or use commandline:
To show labels:
```
curl http://localhost:2005/list_label
``` 
To classify title:
```
curl -X POST -H "Content-Type: application/json" -d '{"text":"Example news title"}' http://localhost:2005/classify
```
## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Peft Library](https://github.com/google-research/peft)
