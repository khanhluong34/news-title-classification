from fastapi import FastAPI, HTTPException, APIRouter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils import load_from_checkpoint

app = FastAPI()
router = APIRouter()

# Load model and tokenizer
checkpoint_path = "./checkpoints/gpt2_news_cls" # Put your model name or path here
model, tokenizer = load_from_checkpoint(checkpoint_path)

# Labels (example)
# b = business, t = science and technology, e = entertainment, m = health
labels = ["business", "entertainment", "health", "science and technology"]

@router.get("/")
def root():
    return {"message": "News classification based on the title"}

# List labels endpoint
@router.get('/list_label')
def list_labels():
    return {'labels': labels}

# Classify endpoint
@router.post('/classify')
def classify_text(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predicted label and probability
    predicted_label = labels[torch.argmax(logits)]
    prob = torch.softmax(logits, dim=1).max().item()
    
    return {'text': text, 'predicted_label': predicted_label, 'prob': prob}
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=2005)
