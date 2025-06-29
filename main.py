from gcd.config import Config
from gcd.model import ModelManager
from gcd.processors import NoOpProcessor, SyncodeProcessor
from gcd.reporter import Reporter
from parserState import GrammarGuidedLLM, ParserStateExtractor
from gcd.tokenScorer import DataCollectingGrammarGuidedLLM
from pathlib import Path
from typing import List, Iterable, Dict
import torch
import pandas as pd
from pathlib import Path
import json
import gc

def main():
    print("Starting training data collection...")

    # Load grammar
    with open("grammars/json.lark", 'r') as f:
        grammar_text = f.read()

    # Load dataset
    with open("jsonprompts.txt", 'r') as f:
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

    all_features = []
    all_targets = []

    for text_idx, text in enumerate(texts):
        print(f"Processing text {text_idx+1}/{len(texts)}: {text[:50]}...")

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
            for dp in data_points:
                parser_state = torch.tensor(dp['parser_state'], dtype=torch.float32)
                syncode_logprobs = torch.tensor(dp['syncode_logprobs'], dtype=torch.float32)
                baseline_logprobs = torch.tensor(dp['baseline_logprobs'], dtype=torch.float32)

                x = torch.cat([parser_state, syncode_logprobs])
                y = baseline_logprobs

                all_features.append(x)
                all_targets.append(y)

            ggllm.reset()

            # Save every 20 texts
            if (text_idx + 1) % 100 == 0:
                data_dict = {
                    "X": torch.stack(all_features),
                    "Y": torch.stack(all_targets),
                }
                torch.save(data_dict, output_path)
                print(f"Saved {len(all_features)} samples after {text_idx+1} texts")

                # Clear memory
                all_features.clear()
                all_targets.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing text {text_idx}: {e}")
            continue

    # Save final batch
    if all_features:
        data_dict = {
            "X": torch.stack(all_features),
            "Y": torch.stack(all_targets),
        }
        torch.save(data_dict, output_path)
        print(f"Final save: {len(all_features)} samples")

if __name__ == "__main__":
    main()