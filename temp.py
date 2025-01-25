from tokenization_bitnet import BitnetTokenizer

# Load the pre-trained tokenizer
tokenizer = BitnetTokenizer.from_pretrained(".")

# Print the tokenizer's vocabulary size
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")