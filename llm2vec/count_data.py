from transformers import AutoTokenizer
import pandas as pd

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/users/neob/Llama-3.1-8B-Instruct")

# Sample some reports to analyze
corpus_df = pd.read_csv("data/pretraining/mntp_corpus_train.csv")
sample_size = min(1000, len(corpus_df))
sample_texts = corpus_df['text'].sample(n=sample_size, random_state=42)

# Tokenize and get lengths
lengths = []
for text in sample_texts:
    tokens = tokenizer(text, truncation=False)
    lengths.append(len(tokens['input_ids']))

# Calculate statistics
import numpy as np
print(f"Token length statistics (sample of {sample_size} reports):")
print(f"  Mean: {np.mean(lengths):.0f} tokens")
print(f"  Median: {np.median(lengths):.0f} tokens")
print(f"  95th percentile: {np.percentile(lengths, 95):.0f} tokens")
print(f"  99th percentile: {np.percentile(lengths, 99):.0f} tokens")
print(f"  Max: {max(lengths)} tokens")

# Show distribution
print("\nLength distribution:")
print(f"  < 256 tokens: {sum(1 for l in lengths if l < 256)} reports")
print(f"  256-512 tokens: {sum(1 for l in lengths if 256 <= l < 512)} reports")
print(f"  512-1024 tokens: {sum(1 for l in lengths if 512 <= l < 1024)} reports")
print(f"  1024-2048 tokens: {sum(1 for l in lengths if 1024 <= l < 2048)} reports")
print(f"  > 2048 tokens: {sum(1 for l in lengths if l >= 2048)} reports")
