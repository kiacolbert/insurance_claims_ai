# load_data.py - Load in batches
from datasets import load_dataset
import json

dataset = load_dataset("deccan-ai/insuranceQA-v2")

# Take only 100 samples
documents = []
for i, item in enumerate(dataset['train']):
    if i >= 100:
        break
    documents.append({
        'content': item['output'],
        'question': item['input']
    })

# Save smaller file
with open('insurance_docs.json', 'w') as f:
    json.dump(documents, f, indent=2)

print(f"✓ Saved {len(documents)} documents")