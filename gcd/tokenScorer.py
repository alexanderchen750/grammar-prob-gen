import torch
import torch.nn.functional as F
from parserState.GrammarGuidedLLM import GrammarGuidedLLM
from typing import List, Dict, Any
import numpy as np


class DataCollectingGrammarGuidedLLM(GrammarGuidedLLM):
    """Extends GrammarGuidedLLM to collect probability data"""
    
    def process_instance_with_probabilities(
        self, 
        text: str,
        model_manager,
        baseline_processor,
        syncode_processor,
        prompt_length: int = 0
    ) -> List[Dict]:
        """
        Use existing process_instance logic but also collect probabilities
        Skip the first `prompt_length` tokens to focus on generation if prompt given.
        TODO/NOTE: Should remove prompt_length as parser_result doesn't have it
        """
        print("Processing instance with probabilities...")
        # Get the parser states using parent class method
        parser_results = self.process_instance(text)
        print("Model forwarding...")
        # Get model predictions
        ids = model_manager.encode(text)
        logits = model_manager.forward(ids)
        
        # Merge probability data with parser states
        training_data = []
        
        for i, parser_result in enumerate(parser_results):
            if i < prompt_length:
                continue

            if i == 0:
                prev_ids = ids[:, :0]
                logits_i = logits[0, 0]
            else:
                prev_ids = ids[:, :i]
                logits_i = logits[0, i - 1]

            # Get baseline and syncode logits for this position
            with torch.no_grad():
                baseline_logits = baseline_processor.process(prev_ids, logits_i.unsqueeze(0))[0]
                syncode_logits = syncode_processor.process(prev_ids, logits_i.unsqueeze(0))[0]

            baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)
            syncode_logprobs = F.log_softmax(syncode_logits, dim=-1)

            data_point = {
                'baseline_logprobs': baseline_logprobs.detach().cpu().numpy(), 
                'syncode_logprobs': syncode_logprobs.detach().cpu().numpy(),
                'parser_state': parser_result.get('onehot_current_state', []),
                'position': i
            }
            training_data.append(data_point)
            del baseline_logits, syncode_logits, baseline_logprobs, syncode_logprobs
            torch.cuda.empty_cache()
        return training_data


    


