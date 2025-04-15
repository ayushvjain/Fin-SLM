import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import sentencepiece as spm
import numpy as np
from torch.utils.data import DataLoader

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

        # Tokenize the chunk text
        input_encoding = self.tokenizer(chunk_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        # Tokenize the tokenized text
        tokenized_encoding = self.tokenizer(tokenized_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),  # Remove extra batch dimension
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': tokenized_encoding['input_ids'].squeeze()
        }



# Load SentencePiece model (trained tokenizer)
sp = spm.SentencePieceProcessor(model_file='financial_tokenizer.model')

# Create Hugging Face tokenizer wrapper using the SentencePiece model
class SPTokenizer:
    def __init__(self, sp_model):
        self.sp_model = sp_model
    
    def __call__(self, text, truncation=True, padding='max_length', max_length=512, return_tensors="pt"):
        # Tokenize the text with SentencePiece
        tokens = self.sp_model.encode(text, out_type=str)
        
        # Convert tokens to input IDs
        input_ids = torch.tensor(self.sp_model.encode(text), dtype=torch.long).unsqueeze(0)  # Batch size 1
        
        # Create attention mask
        attention_mask = (input_ids != self.sp_model.pad_id()).long()
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Initialize the tokenizer with the SentencePiece model
tokenizer = SPTokenizer(sp)

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
    save_total_limit=2
)

model_name = "t5-small"  # You can change this to 't5-base' or 't5-large' if needed
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load your dataset with chunked text and tokenized text
def collate_fn(batch):
    # Remove None values from the batch (entries returned as None)
    batch = [item for item in batch if item is not None]
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }
# Create DataLoader
train_dataset = FinancialDataset(
    chunked_file='financial_corpus_with_chunking.txt',
    tokenized_file='tokenized_output.txt',
    tokenizer=tokenizer
)

#train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # This will automatically use the collate_fn defined above
    eval_dataset=train_dataset,   # Optional: Use a separate validation dataset
)

trainer.train()

model.save_pretrained('./t5_finetuned_model')
tokenizer.save_pretrained('./t5_finetuned_model')
