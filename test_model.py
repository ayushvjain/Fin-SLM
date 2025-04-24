# This file will test the model that we created by running the model.py file.
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_gpu():
    """Check for GPU support using DirectML (for Intel Iris Xe) or CUDA."""
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("Using DirectML device:", dml_device)
        return "directml", dml_device
    except ImportError:
        if torch.cuda.is_available():
            print(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
            return "cuda", torch.device("cuda:0")
        else:
            print("GPU not detected. Using CPU.")
            return "cpu", torch.device("cpu")

def load_model(model_path, device):
    """Load the tokenizer and model from the specified path and move the model to the device."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    return tokenizer, model

def generate_text(tokenizer, model, prompt, max_length=200, temperature=0.7, num_return_sequences=1):
    """Generate text from the model given an input prompt."""
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Test the fine-tuned financial model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model directory (e.g. ./models/run_1_financial_text_finetune/20230101_123456/gpt_financial_model)")
    parser.add_argument("--prompt", type=str, default="What are the current trends in the financial market?",
                        help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation (controls randomness)")
    args = parser.parse_args()
    
    # Determine the appropriate device (DirectML, CUDA, or CPU)
    backend, device = check_gpu()
    
    print(f"\nLoading model from {args.model_path} on device: {device}")
    tokenizer, model = load_model(args.model_path, device)
    print("Model loaded successfully.\n")
    
    # Generate text from the prompt
    print("Input Prompt:")
    print(args.prompt)
    print("\nGenerated Text:")
    output_text = generate_text(tokenizer, model, args.prompt, args.max_length, args.temperature)
    print(output_text)

if __name__ == "__main__":
    main()
