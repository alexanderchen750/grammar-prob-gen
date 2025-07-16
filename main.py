from gcd.config import Config
from gcd.model import ModelManager
from gcd.processors import NoOpProcessor, SyncodeProcessor
from gcd.reporter import Reporter
from parserState import GrammarGuidedLLM, ParserStateExtractor
from gcd.tokenScorer import DataCollectingGrammarGuidedLLM
from pathlib import Path
from typing import List, Iterable, Dict
import torch
from pathlib import Path
import json
import gc
import sys
import pandas as pd


def main():
    print("Starting training data collection...")

    # Load grammar
    with open("grammars/gad.lark", 'r') as f:
        grammar_text = f.read()

    # Load dataset
    with open("gadprompts.txt", 'r') as f:
        texts = [line.strip() for line in f if line.strip()]

    # Initialize model and processors
    model_mgr = ModelManager(Config.model_name)
    baseline_proc = NoOpProcessor()
    syncode_proc = SyncodeProcessor(Config.grammar_path, model_mgr.tokenizer)
    ggllm = DataCollectingGrammarGuidedLLM(
        grammar_text=grammar_text,
        llm_tokenizer_name=Config.model_name,
    )

    # Setup output
    output_path = Path("training_data/grammar_data.pt")
    output_path.parent.mkdir(exist_ok=True)

    all_data_points = []

    for text_idx, text in enumerate(texts):
        print(f"Processing text {text_idx + 1}/{len(texts)}: {text[:50]}...")

        try:
            # Collect data for this text
            data_points = ggllm.process_instance_with_probabilities(
                text=text,
                model_manager=model_mgr,
                baseline_processor=baseline_proc,
                syncode_processor=syncode_proc,
                prompt_length=0
            )

            # Process data points
            # new: just stash the raw arrays/lists into our list of dicts

            for dp in data_points:
                all_data_points.append({
                    "parser_state_onehot": dp['parser_state_onehot'],
                    "syncode_logprobs": dp['syncode_logprobs'],
                    "baseline_logprobs": dp['baseline_logprobs'],
                    "parser_state": dp['parser_state'],
                    "remainder": dp['remainder'],
                    "full_remainder": dp['full_remainder'],
                    "prefix_text": dp['prefix_text'],
                    "next_token": dp['next_token'],
                    "stack": dp['stack'],
                    "value_stack": dp['value_stack'],
                })

            ggllm.reset()

            # Save every 20 texts
            if (text_idx + 1) % 100 == 0:

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing text {text_idx}: {e}")
            ggllm.reset()
            continue

    # Save final batch
    if all_data_points:
        df = pd.DataFrame(all_data_points)
        print(df.head())
        df.to_pickle("training_data/grammar_data_df.pkl")
        print(f"Saved DataFrame with {len(df)} rows to pickle")
        print(f"Final save: {len(all_data_points)} samples")


def parser_sanity_test():
    """Wrapper function to run the parser state consistency test"""
    print("\n=== RUNNING PARSER STATE CONSISTENCY TEST ===")

    with open("grammars/json.lark", 'r') as f:
        grammar_text = f.read()

    test_inputs = [
        '{"name": "Alice", "age": 30}',
        '{"user": {"name": "Bob", "active": true}}',
        '[1, 2, {"key": "value"}]',
        '{"name": "Complex", "nested": {"array": [1, 2, {"deep": null}]}}'
    ]

    result = ParserStateExtractor.test_parser_state_consistency(grammar_text, test_inputs)
    return result


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--test-parser":
        success = parser_sanity_test()
        sys.exit(0 if success else 1)
    else:
        main()
