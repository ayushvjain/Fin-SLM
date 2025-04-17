# import torch
# import sentencepiece as spm
# from transformers import GPT2LMHeadModel
# import os

# def generate_text(prompt, model_path="./gpt_financial_model", max_length=100, temperature=0.7, top_k=50):
#     """Generates text from the trained GPT model based on a given prompt."""
#     print("🔹 Loading model and tokenizer...")
    
#     # Check if model and tokenizer exist
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model directory '{model_path}' does not exist.")
    
#     tokenizer_path = os.path.join(model_path, "tokenizer.model")
#     if not os.path.exists(tokenizer_path):
#         raise FileNotFoundError(f"SentencePiece tokenizer not found at {tokenizer_path}")
    
#     # Load SentencePiece tokenizer
#     sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
#     # Load model
#     try:
#         model = GPT2LMHeadModel.from_pretrained(model_path)
#         model.eval()  # Set to evaluation mode
#     except Exception as e:
#         raise RuntimeError(f"Failed to load model: {e}")
    
#     print("🔹 Tokenizing input prompt...")
#     # Tokenize the prompt using SentencePiece directly
#     tokens = sp.encode(prompt, out_type=str)
#     token_ids = [sp.piece_to_id(token) for token in tokens]
    
#     # Convert to tensor
#     input_ids = torch.tensor([token_ids], dtype=torch.long)
    
#     print(f"🔹 Generating text with prompt: '{prompt}'")
#     # Generate text
#     with torch.no_grad():
#         output_sequences = model.generate(
#             input_ids,
#             max_length=max_length,
#             temperature=temperature,
#             top_k=top_k,
#             pad_token_id=3,  # Use the <pad> token ID
#             bos_token_id=1,  # Use the <s> token ID
#             eos_token_id=2,  # Use the </s> token ID
#             do_sample=True,
#             num_return_sequences=1
#         )
    
#     # Decode the generated token IDs back to text using SentencePiece
#     generated_ids = output_sequences[0].tolist()
#     generated_tokens = [sp.id_to_piece(token_id) for token_id in generated_ids]
#     generated_text = "".join(generated_tokens).replace("▁", " ")  # Replace SentencePiece space character
    
#     print("🔹 Text generation complete.")
#     return generated_text

# def check_model_structure(model_path="./gpt_financial_model"):
#     """Helper function to check and print the structure of the model directory."""
#     if not os.path.exists(model_path):
#         print(f"❌ Model directory '{model_path}' does not exist.")
#         return False
    
#     print(f"📁 Model directory '{model_path}' exists.")
#     files = os.listdir(model_path)
#     print(f"📋 Files in model directory: {files}")
    
#     if "pytorch_model.bin" not in files:
#         print("❌ Missing pytorch_model.bin file.")
#         return False
    
#     if "config.json" not in files:
#         print("❌ Missing config.json file.")
#         return False
    
#     if "tokenizer.model" not in files:
#         print("❌ Missing tokenizer.model (SentencePiece) file.")
#         return False
    
#     print("✅ All essential model files found.")
#     return True

# if __name__ == "__main__":
#     model_path = "./gpt_financial_model"
    
#     # Check model structure first
#     print("🔍 Checking model structure...")
#     model_ok = check_model_structure(model_path)
    
#     if not model_ok:
#         print("⚠️ Model structure issues detected. Running with caution...")
    
#     # Example usage
#     try:
#         prompt = "The future of the stock market in 2025 is expected to"
#         print(f"🚀 Generating text for prompt: '{prompt}'")
#         generated_output = generate_text(prompt, model_path=model_path)
#         print("\n📝 Generated Text:\n" + "="*40 + "\n" + generated_output + "\n" + "="*40)
#     except Exception as e:
#         import traceback
#         print(f"❌ Error during text generation: {e}")
#         traceback.print_exc()

import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def check_gpu():
    """Check for GPU support using DirectML (for Intel Iris Xe) or CUDA."""
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("✅ Using DirectML device:", dml_device)
        return "directml", dml_device
    except ImportError:
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            return "cuda", torch.device("cuda:0")
        else:
            print("⚠️ GPU not detected. Using CPU.")
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
    print("✅ Model loaded successfully.\n")
    
    # Generate text from the prompt
    print("Input Prompt:")
    print(args.prompt)
    print("\nGenerated Text:")
    output_text = generate_text(tokenizer, model, args.prompt, args.max_length, args.temperature)
    print(output_text)

if __name__ == "__main__":
    main()
