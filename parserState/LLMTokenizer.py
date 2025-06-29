from transformers import AutoTokenizer

class LLMTokenizer:
    """Wrapper for LLM tokenizers to provide a consistent interface."""
    
    def __init__(self, tokenizer_name="Qwen/Qwen3-4B"):
        """Initialize with a model name or tokenizer instance."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def encode(self, text):
        """Convert text to token IDs."""
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        if isinstance(token_ids, int):
            return self.tokenizer.decode([token_ids])
        return self.tokenizer.decode(token_ids)
    
    def encode_with_details(self, text):
        """Encode text and return token IDs and their string representations."""
        token_ids = self.encode(text)
        token_strings = [self.decode([id]) for id in token_ids]
        return token_ids, token_strings