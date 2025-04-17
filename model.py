import os
import time
import torch
import json
import math
from datetime import datetime
import traceback
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_from_disk, concatenate_datasets


def check_gpu():
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("✅ Using DirectML device:", dml_device)
        return "directml"
    except ImportError:
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            print("⚠️ GPU not detected. Using CPU.")
            return "cpu"

class FinancialTextTrainer:
    def __init__(self, run_id, params, base_dir="./models"):
        """Initialize the financial text trainer with parameters."""
        self.params = params
        self.run_id = run_id

        # Create a unique run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"run_{run_id}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Save parameters for future reference
        with open(os.path.join(self.run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        # Set the file containing financial text data
        self.text_file = params.get("text_file", "financial_corpus_without_chunking.txt")
        self.max_seq_length = params.get("max_seq_length", 512)

        # Determine which GPU backend to use: DirectML, CUDA, or CPU.
        self.gpu_backend = check_gpu()
        if self.gpu_backend == "directml":
            import torch_directml
            self.device = torch_directml.device()
        elif self.gpu_backend == "cuda":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.use_gpu = self.gpu_backend in ["directml", "cuda"]
        self.use_mixed_precision = params.get("use_mixed_precision", True) and self.use_gpu

        print(f"🔹 Initialized run {run_id} with output directory: {self.run_dir}")
        print(f"🔹 Using GPU: {'Yes' if self.use_gpu else 'No'}")
        print(f"🔹 Device: {self.device}")
        print(f"🔹 Mixed Precision: {'Yes' if self.use_mixed_precision else 'No'}")

    def load_and_preprocess_dataset(self):
        print("🔹 Loading text data from file...")
        try:
            with open(self.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            raise IOError(f"Error loading text file {self.text_file}: {e}")

        # Split the text into paragraphs (adjust the delimiter as needed)
        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
        if not paragraphs:
            raise ValueError("No valid text examples found in the file.")

        dataset = Dataset.from_dict({"text": paragraphs})
        print(f"🔹 Loaded dataset with {len(dataset)} examples")
        return dataset

    def tokenize_function(self, examples, tokenizer):
        """Tokenize the text examples using the provided tokenizer."""
        return tokenizer(examples["text"], truncation=True, max_length=self.max_seq_length)

    def train_gpt_model(self):
        """Load a pre-trained GPT-2 model and fine-tune it on the financial text dataset."""
        model_name = self.params.get("model_name", "gpt2")
        print(f"🔹 Loading pre-trained model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 does not define a pad token by default. Set it to the EOS token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(self.device)

        # Load and preprocess the dataset
        dataset = self.load_and_preprocess_dataset()
        print("🔹 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda x: self.tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=["text"]
        )

        if len(tokenized_dataset) < 2:
            print("⚠️ Dataset too small for splitting. Using full dataset for both training and evaluation.")
            train_dataset = tokenized_dataset
            eval_dataset = tokenized_dataset
        else:
            split_dataset = tokenized_dataset.train_test_split(test_size=self.params.get("eval_split", 0.05))
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]

        print(f"🔹 Train dataset size: {len(train_dataset)}")
        print(f"🔹 Eval dataset size: {len(eval_dataset)}")

        # Use the data collator for language modeling (no masking for causal LM)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.run_dir, "gpt_financial_model"),
            eval_steps=self.params.get("eval_steps", 500),
            save_strategy="steps",
            save_steps=self.params.get("save_steps", 500),
            logging_steps=self.params.get("logging_steps", 100),
            num_train_epochs=self.params.get("epochs", 3),
            per_device_train_batch_size=self.params.get("batch_size", 4),
            per_device_eval_batch_size=self.params.get("batch_size", 4),
            learning_rate=self.params.get("learning_rate", 5e-5),
            weight_decay=self.params.get("weight_decay", 0.01),
            gradient_accumulation_steps=self.params.get("gradient_accumulation_steps", 1),
            fp16=self.use_mixed_precision,
            report_to="none",
            logging_dir=os.path.join(self.run_dir, "logs"),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        print("🚀 Starting fine-tuning...")
        start_time = time.time()
        trainer.train()
        total_time = time.time() - start_time
        print(f"✅ Fine-tuning complete in {total_time:.2f} seconds.")

        # Evaluate the model on the evaluation dataset and compute perplexity
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss", None)
        if eval_loss is not None:
            perplexity = math.exp(eval_loss)
            print(f"🔹 Evaluation Loss: {eval_loss:.4f}")
            print(f"🔹 Perplexity: {perplexity:.4f}")

        # Save the final model and tokenizer
        model_save_path = os.path.join(self.run_dir, "gpt_financial_model")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        # Save final training parameters for reference
        with open(os.path.join(model_save_path, "final_params.json"), "w") as f:
            json.dump(self.params, f, indent=4)

        print(f"✅ Model and tokenizer saved at {model_save_path}")
        return model_save_path

    def run_pipeline(self):
        """Run the complete fine-tuning pipeline."""
        print(f"\n{'='*80}")
        print(f"🚀 Starting run {self.run_id} with parameters:")
        for key, value in self.params.items():
            print(f"   {key}: {value}")
        print(f"{'='*80}\n")

        try:
            model_save_path = self.train_gpt_model()
            print(f"\n{'='*80}")
            print(f"✅ Run {self.run_id} completed successfully!")
            print(f"   Model artifacts saved to: {model_save_path}")
            print(f"{'='*80}\n")
            return True
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"Error in run {self.run_id}: {e}")
            traceback.print_exc()
            print(f"{'='*80}\n")
            with open(os.path.join(self.run_dir, "error_log.txt"), "w") as f:
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
            return False

def main():
    # Use DirectML on Intel Iris Xe GPU if available, otherwise fallback.
    gpu_backend = check_gpu()
    has_gpu = gpu_backend in ["directml", "cuda"]
    parameter_sets = [
        {
            "name": "financial_text_finetune_C",
            "text_file": "financial_corpus_without_chunking.txt",
            "max_seq_length": 512,
            "epochs": 30,
            "batch_size": 4,
            "learning_rate": 4e-5,
            "gradient_accumulation_steps": 2,
            "weight_decay": 0.01,
            "eval_split": 0.05,
            "eval_steps": 800,
            "save_steps": 800,
            "logging_steps": 100,
            "model_name": "gpt2",
            "use_gpu": has_gpu,
            "use_mixed_precision": has_gpu
        }
    ]

    base_dir = "./models"
    os.makedirs(base_dir, exist_ok=True)

    results = []
    for i, params in enumerate(parameter_sets):
        run_id = f"{i+1}_{params['name']}"
        print(f"\n{'#'*100}")
        print(f"## Starting Run {run_id}: {params['name']}")
        print(f"{'#'*100}\n")

        trainer = FinancialTextTrainer(run_id, params, base_dir)
        success = trainer.run_pipeline()

        results.append({
            "run_id": run_id,
            "name": params["name"],
            "success": success,
            "output_dir": trainer.run_dir,
            "used_gpu": trainer.use_gpu
        })

    print(f"\n{'#'*100}")
    print("## SUMMARY OF ALL RUNS")
    print(f"{'#'*100}\n")
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        gpu_info = "Used GPU" if result["used_gpu"] else "Used CPU"
        print(f"{status} - Run {result['run_id']}: {result['name']} ({gpu_info})")
        print(f"  Output directory: {result['output_dir']}\n")

    summary_file = os.path.join(base_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()