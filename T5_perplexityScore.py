import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
from torch.nn import functional as F

# Load model and tokenizer (checkpoint-260000)
model = T5ForConditionalGeneration.from_pretrained('./results/checkpoint-260000')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Set the decoder_start_token_id to the pad_token_id
model.config.decoder_start_token_id = model.config.pad_token_id  # Set to pad_token_id

# Tokenize the prompt
prompt = "What are the latest trends in global finance?"
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass to get loss
outputs = model(**inputs, labels=inputs['input_ids'])
logits = outputs.logits

# Calculate cross-entropy loss
shift_labels = inputs['input_ids'][:, 1:].contiguous()
shift_logits = logits[:, :-1, :].contiguous()

# Cross-entropy loss
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# Perplexity
perplexity = torch.exp(loss)
print(f"Perplexity: {perplexity.item()}")
