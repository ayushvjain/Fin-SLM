import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset

# Dataset class to handle chunked and tokenized text
class FinancialDataset(Dataset):
    def __init__(self, chunked_file, tokenized_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read chunked text
        with open(chunked_file, 'r', encoding='utf-8') as file:
            self.chunks = file.readlines()
        
        # Read tokenized text
        with open(tokenized_file, 'r', encoding='utf-8') as file:
            self.tokenized_chunks = file.readlines()

        # Ensure both datasets have the same length
        assert len(self.chunks) == len(self.tokenized_chunks), \
            f"Mismatch between chunked text and tokenized text lengths: {len(self.chunks)} vs {len(self.tokenized_chunks)}"
        
        # Filter out invalid entries (empty or malformed)
        valid_indices = [
            idx for idx in range(len(self.chunks))
            if self.chunks[idx].strip() and self.tokenized_chunks[idx].strip()
        ]
        self.chunks = [self.chunks[i] for i in valid_indices]
        self.tokenized_chunks = [self.tokenized_chunks[i] for i in valid_indices]

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        # Get the chunk text
        chunk_text = self.chunks[idx]
        tokenized_text = self.tokenized_chunks[idx]

        # Tokenize the chunk text and tokenized text
        input_encoding = self.tokenizer(chunk_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        tokenized_encoding = self.tokenizer(tokenized_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),  # Remove extra batch dimension
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': tokenized_encoding['input_ids'].squeeze()
        }

# Load the pretrained T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Initialize the T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Prepare the dataset
train_dataset = FinancialDataset(
    chunked_file='financial_corpus_with_chunking.txt',
    tokenized_file='tokenized_output.txt',
    tokenizer=tokenizer
)

# Prepare the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=10_000,
    save_total_limit=2,
    # Enable GPU if CUDA is available
    no_cuda=False if torch.cuda.is_available() else True,
    # Automatically choose the best device (GPU or CPU)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Define the data collator for seq2seq tasks (handles padding)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Pass the dataset directly
    eval_dataset=train_dataset,   # Optional: Use a separate validation dataset if available
    data_collator=data_collator   # Pass the data collator to handle padding dynamically
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./t5_finetuned_model')
tokenizer.save_pretrained('./t5_finetuned_model')
