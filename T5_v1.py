import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import Dataset
import os

# === Dataset Class ===
class FinancialDataset(Dataset):
    def __init__(self, chunked_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load text chunks
        with open(chunked_file, 'r', encoding='utf-8') as f:
            self.chunks = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        text = self.chunks[idx]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# === Load custom SentencePiece tokenizer ===
tokenizer = T5Tokenizer.from_pretrained('./', model_max_length=512, legacy=False)
tokenizer.sp_model.Load("financial_tokenizer.model")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"

# === Define T5 config ===
config = T5Config(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    d_ff=2048,
    num_layers=6,
    num_heads=8,
    dropout_rate=0.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.pad_token_id
)

# === Initialize model from scratch ===
model = T5ForConditionalGeneration(config)

# === Load dataset ===
dataset = FinancialDataset("financial_corpus_with_chunking.txt", tokenizer)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir="./t5_scratch_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=10000,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"
)

# === Data collator ===
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Start training ===
trainer.train()

# === Save model ===
model.save_pretrained("./t5_scratch_model_final")
tokenizer.save_pretrained("./t5_scratch_model_final")
