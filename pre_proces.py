import os
import sentencepiece as spm

def chunk_text_file(input_file='financial_corpus_without_chunking.txt', 
                    output_file='financial_corpus_with_chunking.txt', 
                    chunk_size=1000, 
                    overlap=100, 
                    encoding='utf-8'):
    """
    Chunk a large text file into smaller overlapping pieces.
    
    Parameters:
    -----------
    input_file : str
        Path to the input text file
    output_file : str
        Path to save the chunked output
    chunk_size : int
        Size of each chunk
    overlap : int
        Number of characters to overlap between chunks
    encoding : str
        File encoding
    """
    # Validate parameters
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("Chunk size must be positive and overlap must be non-negative")
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    
    # Read the entire file
    try:
        with open(input_file, 'r', encoding=encoding) as file:
            text = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    except Exception as e:
        raise IOError(f"Error reading the input file: {e}")
    
    # Chunk and write the text
    with open(output_file, 'w', encoding=encoding) as outfile:
        for start in range(0, len(text), chunk_size - overlap):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                outfile.write(chunk + "\n\n")
    
    print(f"Chunking complete. Output written to {output_file}")

def train_sentencepiece_model(input_file='financial_corpus_with_chunking.txt', 
                               model_prefix='financial_tokenizer', 
                               vocab_size=8000, 
                               model_type='unigram'):
    """
    Train a SentencePiece tokenizer model.
    
    Parameters:
    -----------
    input_file : str
        Path to the input text file
    model_prefix : str
        Prefix for the output model files
    vocab_size : int
        Size of the vocabulary
    model_type : str
        Type of tokenization model (unigram, bpe, etc.)
    
    Returns:
    --------
    str
        Path to the trained model
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    # Financial domain-specific user-defined symbols
    financial_symbols = [
        # Currencies
        'USD,EUR,JPY,CNY,AUD,INR,GBP,Bitcoin,ETH',
        # Financial terms
        'dividend,capital_gain,interest,income,asset,liability',
        # Financial instruments
        'stock,share,bond,IPO,SPAC,capital,portfolio',
        # Market indices
        'NASDAQ,NYSE,FTSE,HangSeng',
        # Additional financial indices
        'ETF,REIT,SP500,NASDAQ100'
    ]
    
    # SentencePiece training parameters
    training_params = [
        f'--input={input_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type={model_type}',
        '--character_coverage=1.0',
        '--normalization_rule_name=identity',
        f'--user_defined_symbols={",".join(financial_symbols)}',
        '--max_sentencepiece_length=16', 
        '--split_by_whitespace=true',
        '--control_symbols=<s>,</s>,<pad>',
        '--add_dummy_prefix=false'
    ]
    
    # Train the model
    spm.SentencePieceTrainer.train(' '.join(training_params))
    
    print(f"SentencePiece model trained and saved as {model_prefix}.model")
    return f'{model_prefix}.model'

def tokenize_text(model_path, input_file='financial_corpus_with_chunking.txt', output_file='tokenized_output.txt'):
    """
    Tokenize text using the trained SentencePiece model.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained SentencePiece model
    input_file : str
        Path to the input text file to tokenize
    output_file : str
        Path to save the tokenized output
    """
    # Load the SentencePiece model
    sp = spm.SentencePieceProcessor(model_file=model_path)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        text = infile.read()
    
    # Tokenize and write output
    with open(output_file, 'w', encoding='utf-8') as outfile:
        chunks = text.split('--- Chunk')
        
        for i, chunk in enumerate(chunks[1:], 1):
            # Extract chunk header and text
            lines = chunk.split('\n')
            chunk_header = lines[0]
            chunk_text = '\n'.join(lines[1:]).strip()
            
            # Tokenize the chunk
            tokens = sp.encode(chunk_text, out_type=str)
            
            # Write tokenized chunk
            outfile.write(f"--- Chunk {i} {chunk_header}\n")
            outfile.write(' '.join(tokens) + '\n\n')
    
    print(f"Tokenization complete. Output written to {output_file}")

def main():
    try:
        # Chunk the input text file
        chunk_text_file(
            input_file='financial_corpus_without_chunking.txt', 
            output_file='financial_corpus_with_chunking.txt', 
            chunk_size=1000,
            overlap=100
        )
        
        # Train SentencePiece model
        model_path = train_sentencepiece_model(
            input_file='financial_corpus_with_chunking.txt',
            model_prefix='financial_tokenizer',
            vocab_size=8000,
            model_type='unigram'
        )
        
        # Tokenize text
        tokenize_text(
            model_path=model_path,
            input_file='financial_corpus_with_chunking.txt',
            output_file='tokenized_output.txt'
        )
        
        # Print vocabulary preview
        sp = spm.SentencePieceProcessor(model_file=model_path)
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print("\nVocabulary Preview:")
        print(vocab[:50])  # Print first 50 tokens
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()