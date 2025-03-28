import sentencepiece as spm

# Load the trained tokenizer
sp = spm.SentencePieceProcessor(model_file="finance_tokenizer.model")

# Test tokenization
text = "The stock price of MSFT increased by 5% today, while the <market_index> fell by 2%."
tokens = sp.encode(text, out_type=str)
print("Tokenized text:", tokens)

# Decode tokens back to text
decoded_text = sp.decode(tokens)
print("Decoded text:", decoded_text)