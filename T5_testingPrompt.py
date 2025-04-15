from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./results/checkpoint-260000')
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)  # Use the appropriate tokenizer

# Ensure decoder_start_token_id is set
model.config.decoder_start_token_id = model.config.pad_token_id  # Set the decoder start token to pad token id

# Define the prompt
prompt = input("Ask a question :")

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output using the model
output = model.generate(inputs['input_ids'], max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# Decode the output
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Generated response: {response}")