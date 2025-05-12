from transformers import BertTokenizer, BertModel
import torch
import os

# Initialize the BERT tokenizer and model for embedding conversion
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def convert_text_to_embeddings(text: str) -> list[float]:
    """
    Converts text into a vector embedding using BERT model.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
    return embeddings

def read_file(file_path):
    """
    Reads any text file and returns its content as a string.
    """
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     return file.read()
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

