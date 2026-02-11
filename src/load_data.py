# load_data.py
import os
from datasets import load_dataset
import json

# Load dataset
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
dataset = load_dataset("deccan-ai/insuranceQA-v2", token=HF_TOKEN)
print(f"Dataset loaded: {dataset}")
# Convert to simple format
documents = []
for item in dataset['train'][:100]:  # Start with 100 for testing
    parsed = json.loads(item)
    documents.append({
        'content': parsed['output'],
        'question': parsed['input']
    })

# Save for testing
with open('insurance_docs.json', 'w') as f:
    json.dump(documents, f, indent=2)

print(f"Loaded {len(documents)} documents")
print(f"Sample: {documents[0]['question'][:100]}")
