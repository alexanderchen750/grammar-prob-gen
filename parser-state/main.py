from GrammarGuidedLLM import GrammarGuidedLLM
from ParserStateExtractor import ParserStateExtractor
import json

def main(output_file):
    # File paths - manually change these as needed
    grammar_file = "SQL.lark"
    dataset_file = "SQL_sample.txt"
    #output_file = "results1.txt"

    # Fallback dataset if the file can't be opened
    
    
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
    #print(f"Processing {len(dataset)} examples...")
    results = builder.process_dataset(dataset)
    
    # Write results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results written to {output_file}")
    #print(f"Total errors encountered: {error_count}")

if __name__ == "__main__":
    main("results1.txt")
    main("results2.txt")
    #main("results3.txt")
    #main("results3.txt")