import pandas as pd
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/users/neob/Llama-3.1-8B-Instruct")

# Load your training data
df = pd.read_csv("data/finetuning/supervised_pairs_train.csv")

# Analyze token lengths
lengths1 = []
lengths2 = []
combined_lengths = []

for _, row in df.sample(n=min(1000, len(df))).iterrows():
    # Simulate the prompt formatting from MedicalPairData
    query = f"Compare these medical reports for semantic similarity:\n\nReport: {row['sentence1']}"
    doc = f"Report: {row['sentence2']}"
    
    tokens1 = tokenizer.encode(query, add_special_tokens=True)
    tokens2 = tokenizer.encode(doc, add_special_tokens=True)
    
    lengths1.append(len(tokens1))
    lengths2.append(len(tokens2))
    combined_lengths.append(max(len(tokens1), len(tokens2)))

# Print statistics
print(f"Query lengths - Mean: {np.mean(lengths1):.0f}, 95th percentile: {np.percentile(lengths1, 95):.0f}")
print(f"Document lengths - Mean: {np.mean(lengths2):.0f}, 95th percentile: {np.percentile(lengths2, 95):.0f}")
print(f"Max of pair - 95th percentile: {np.percentile(combined_lengths, 95):.0f}")
print(f"\nPercentage truncated at different lengths:")
print(f"256 tokens: {(np.array(combined_lengths) > 256).mean() * 100:.1f}%")
print(f"384 tokens: {(np.array(combined_lengths) > 384).mean() * 100:.1f}%")
print(f"512 tokens: {(np.array(combined_lengths) > 512).mean() * 100:.1f}%")
