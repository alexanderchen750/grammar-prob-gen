from .GrammarGuidedLLM import GrammarGuidedLLM
from .ParserStateExtractor import ParserStateExtractor
import json
import os


"""Main function to process a dataset using a grammar-guided LLM parser.
Reads a grammar file and a dataset file, processes the dataset using the parser,
and writes the results to an output file.
"""
def main(output_file):

    # File paths - manually change these as needed
    this_dir = os.path.dirname(__file__)
    grammar_file = os.path.join(this_dir, "grammars", "SQL.lark")
    dataset_file = os.path.join(this_dir, "SQL_sample.txt")
    
    # Read grammar from file
    with open(grammar_file, 'r') as f:
        grammar = f.read()

    # Read dataset from file
    if dataset_file.endswith(".txt"):
        with open(dataset_file, 'r') as f:
            dataset = [line.strip() for line in f if line.strip()]
    else:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
            # Ensure dataset is a list
            if not isinstance(dataset, list):
                dataset = [dataset]
        dataset = [json.dumps(example) for example in dataset]
    # Initialize parser
    builder = GrammarGuidedLLM(
        grammar_text=grammar, 
        llm_tokenizer_name="gpt2", 
        stack_context_length=3, 
    )
    """
    extractor = ParserStateExtractor(grammar)
    results = []
    error_count = 0
    for i, instance in enumerate(dataset):
        print(f"Processing instance {i+1}")
        extractor.reset()
        try:
            result = extractor.advance_parser(instance, top_k=3)
            results.append(result)
        except Exception as e:
            print(f"Error processing instance {i+1}: {e}")
            error_count += 1
            continue
        """
    # Process dataset
    results = builder.process_dataset(dataset)
    
    # Write results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to {output_file}")
    builder.reset()  # Reset the parser state extractor
    

if __name__ == "__main__":
    main("results1.txt")
    main("results2.txt")