from LLMTokenizer import LLMTokenizer
from ParserStateExtractor import ParserStateExtractor
import json


import copy
from itertools import islice

class GrammarGuidedLLM:
    """Integrates LLM, tokenizers, and parser for grammar-guided generation."""
    
    def __init__(self, grammar_text, llm_tokenizer_name="gpt2", stack_context_length=3, debug=False):
        """Initialize with grammar and tokenizer settings."""
        self.parser_extractor = ParserStateExtractor(grammar_text)
        #self.parser_extractor.save_state_mapping("grammar_mapping.json")
        self.llm_tokenizer = LLMTokenizer(llm_tokenizer_name)
        self.stack_context_length = stack_context_length
        self.debug = debug

    def log(self, *args):
        if self.debug:
            print("[DEBUG]", *args)

    def process_instance(self, text):
        """
        Incrementally step through the LLM tokens, but only call
        `advance_parser()` once we’ve *finished* at least one new
        lexical token.  Any partial token still in flight is kept in
        `remainder`.
        """
        llm_tokens      = self.llm_tokenizer.encode(text)
        lex_result = self.parser_extractor.get_lexical_tokens_with_positions(text)
        if isinstance(lex_result, tuple):
            lex_positions, rem = lex_result
            #print("remaind", rem)
        else:
            lex_positions = lex_result
        results = []

        prefix_text          = ""       # full text emitted so far
        last_lex_end_idx     = 0        # char index we have already parsed up to
        cur_lex_idx          = -1       # last *finished* lexical token
        last_snapshot        = {}       # cached result when no new lex progress

        for i, tok in enumerate(llm_tokens):
            piece        = self.llm_tokenizer.decode([tok])
            prefix_text += piece
            # Lexical progress
            new_lex_idx = cur_lex_idx
            while (new_lex_idx + 1 < len(lex_positions) and
                len(prefix_text) >= lex_positions[new_lex_idx + 1][1]):
                new_lex_idx += 1

            #New lexical token
            if new_lex_idx > cur_lex_idx:
                # We have at least one brand-new, *complete* lexical token.
                chunk_start   = last_lex_end_idx
                chunk_end     = lex_positions[new_lex_idx][1]
                completed     = text[chunk_start:chunk_end]

                result_set = self.parser_extractor.advance_parser(
                    completed, top_k=self.stack_context_length
                )
                remainder   = prefix_text[chunk_end:]             
                result_set["remainder"] = remainder               
                last_snapshot = copy.deepcopy(result_set)

                last_lex_end_idx = chunk_end
                cur_lex_idx      = new_lex_idx
            #No Lexical Token
            else:
                # No lexical progress – clone previous result and extend remainder
                result_set = copy.deepcopy(last_snapshot)
                dangling = prefix_text[last_lex_end_idx:]
                result_set.setdefault("remainder", "")
                result_set["remainder"] += dangling     # overwrite with fresh slice

            #next token
            next_tok_str = (self.llm_tokenizer.decode([llm_tokens[i + 1]])
                            if i < len(llm_tokens) - 1 else None)
            result_set["next_token"] = next_tok_str
            results.append(result_set)

            # Optional verbose trace
            if getattr(self, "verbose", False):
                self.log(f"Prefix text so far: «{prefix_text}»")
                self.log(f"   remainder      : «{result_set['remainder']}»")
        return results

    
    def process_dataset(self, dataset):
        """Process a set of inputs to create training sets."""
        all_instances = []
        
        for instance in dataset:
            # Validate grammar if needed
            try:
                result = self.process_instance(instance)
                all_instances.append(result)
                self.parser_extractor.reset()
            except Exception as e:
                print(f"Error processing instance: {instance}")

                print("Error: ", str(e))
                continue
        return all_instances
