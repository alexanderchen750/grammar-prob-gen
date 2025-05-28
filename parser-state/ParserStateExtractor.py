from lark import Lark
from lark.lexer import Token
import hashlib
import json
from lark.exceptions import UnexpectedCharacters
import os

"""
TODO: Consistency Guarantee
Ensure consistency with the parser's state machine.
Store global state mapping across sessions to ensure consistency
Current solution: cache a parser, which should be consistent across runs

TODO: Error Handling

TODO: Performance
Changing output formats to one hot encoding/tensor or something more efficient?can
"""

class ParserStateExtractor:
    """
    Extracts consistent parser states from partial sequences using Lark's LALR parser.
    Provides stable state IDs and stack elements that remain consistent between runs.
    """
    _global_state_counter = 0
    _global_state_mapping = {} 
    
    def __init__(self, grammar_text, mapping_file=None):
        """
        Initialize with a grammar string
        
        Args:
            grammar_text: The Lark grammar as a string
        """
        # We'll calculate a grammar hash to ensure we detect grammar changes
        grammar_hash = hashlib.md5(grammar_text.encode()).hexdigest()[:8]
    
        self.parser = Lark(grammar_text, parser='lalr', cache=True,)
        self.interactive_parser = self.parser.parse_interactive('')
        self.tokens = []
        self.current_string = ""
        self.current_remainder = ""
        self.debug= False 
        self._state_mapping = self._initialize_state_mappings()
        """
        for state in self.interactive_parser.parser_state.parse_conf.parse_table.states:
            # Get the state ID
            print( self.interactive_parser.parser_state.parse_conf.parse_table.states[state].state_id)
            print(" ")
            # Get the consistent state ID
            # Store the mapping
        """



    def feed_input(self, text):
        """
        Advance the parser with a new text input.
        """
        # Reset the interactive parser
        self.current_string = self.current_string+text
        new_tokens, self.current_remainder = self.get_tokens_with_remainder(self.current_string)
        different_tokens = new_tokens[len(self.tokens):]
        for token in different_tokens:
            self.interactive_parser.feed_token(token)
        self.tokens = new_tokens
        
    def log(self, *args):
        if self.debug:
            print("[DEBUG]", *args)

    def get_tokens(self, sequence):
        """Get all tokens from the sequence"""
        lexer = self.parser.lex(sequence)
        return list(lexer)
    def get_tokens_with_remainder(self, sequence):
        try:
            # Try to tokenize the entire sequence
            tokens = list(self.parser.lex(sequence))
            
            # If we have tokens, check if they cover the entire input
            if tokens:
                last_token_end = tokens[-1].end_pos
                if last_token_end < len(sequence):
                    # There's untokenized content at the end
                    remainder = sequence[last_token_end:]
                    return tokens, remainder
                return tokens, ""  # All content was tokenized successfully
            else:
                # No tokens found at all
                return [], sequence
                
        except UnexpectedCharacters as e:
            # We encountered an error while lexing
            # Get tokens up to the error position
            tokens = list(self.parser.lex(sequence[:e.pos_in_stream]))
            
            # Everything from the error position to the end is our remainder
            remainder = sequence[e.pos_in_stream:]
            return tokens, remainder
        
    def get_lexical_tokens_with_positions(self, text):
        """
        Get all lexical tokens with their positions in the original text 
        reutrn list[tuple(start,end)]
        """
        try:
            lexer = self.parser.lex(text)
            tokens = list(lexer)
            token_info = []
            
            for token in tokens:
                token_info.append((token.start_pos, token.end_pos))
    
            return token_info
            
        except UnexpectedCharacters as e:
            # Process tokens up to the error
            tokens = list(self.parser.lex(text[:e.pos_in_stream]))
            token_info = []
            
            for token in tokens:
                token_info.append((token.start_pos, token.end_pos))
            remainder = text[e.pos_in_stream:]
            return token_info, remainder

    def _initialize_state_mappings(self):
        """Generate consistent state mappings by directly accessing the transition table"""
        parse_table = self.interactive_parser.parser_state.parse_conf.parse_table
        
        # Build fingerprints for all states directly from the transition table
        state_fingerprints = {}
        
        # First pass: collect all state information
        for state_id in sorted(parse_table.states.keys()):
            # Get transitions directly
            transitions = []
            
            # Process action entries (shift/reduce/accept)
            for term, (action, target) in sorted(parse_table.states[state_id].items()):
                term_name = term.name if hasattr(term, 'name') else str(term)
                
                # Create a canonical representation of the action
                # Handle both string name and class instance
                if hasattr(action, '__name__'):
                    action_name = action.__name__
                else:
                    # For newer Lark versions, get the class name
                    action_name = action.__class__.__name__
                    
                if action_name == 'Shift':
                    action_str = 'SHIFT'
                elif action_name == 'Reduce':
                    # Get rule details in a deterministic way
                    rule = target
                    origin = str(rule.origin)
                    expansion = '+'.join(str(x) for x in rule.expansion)
                    action_str = f"REDUCE:{origin}:{expansion}"
                else:
                    action_str = action_name
                    
                transitions.append((term_name, action_str))
            
            # Get lookaheads if available
            if hasattr(parse_table, 'lookaheads') and state_id in parse_table.lookaheads:
                for item, terms in sorted(parse_table.lookaheads[state_id].items()):
                    item_str = str(item)
                    terms_str = '+'.join(sorted(str(t) for t in terms))
                    transitions.append((f"LA:{item_str}", terms_str))
            
            # Sort for determinism
            state_fingerprints[state_id] = tuple(sorted(transitions))
        
        # Create canonical string representations of fingerprints
        fp_strings = {}
        for state_id, fingerprint in state_fingerprints.items():
            parts = []
            for term, action in fingerprint:
                parts.append(f"{term}:{action}")
            fp_strings[state_id] = "||".join(parts)
        
        # Create mapping based on sorted fingerprint strings
        sorted_entries = sorted(fp_strings.items(), key=lambda x: x[1])
        
        # Assign sequential IDs
        state_mapping = {}
        for i, (state_id, _) in enumerate(sorted_entries):
            state_mapping[state_id] = f"S{i:03d}"
        
        return state_mapping

    def _get_consistent_state_id(self, state_id, parse_table):
        """
        Get consistent state ID from pre-computed mapping
        """
        return self._state_mapping.get(state_id, None)
    
    def _get_value_stack(self, value_stack, top_k=3):
        """
        Extract terminal categories from the value stack
        """
        terminal_categories = []
        # Get the last top_k items, but check if there are enough items first
        values_to_process = value_stack[-top_k:] if len(value_stack) >= top_k else value_stack
        
        for value in values_to_process:
            if isinstance(value, Token):
                terminal_categories.append(value.type)  # Use token type not value
            elif hasattr(value, 'data'):  # For Tree objects
                # This is a parse tree, extract its type
                terminal_categories.append(f"{value.data}")
            else:
                terminal_categories.append(type(value).__name__)
                
        return terminal_categories

    def get_parser_state(self, interactive_parser, top_k=3):
        """
        Get the parser state from an interactive parser
        """
        # Get the current parser state
        parser_state = interactive_parser.parser_state
        parse_table = parser_state.parse_conf.parse_table
        # Get raw state information
        raw_state_id = parser_state.position
        raw_stack = list(parser_state.state_stack)
        value_stack = list(parser_state.value_stack)
        #print(value_stack)
        #print(" ")
        # Get the value stack
        value_stack = self._get_value_stack(value_stack, top_k)
        
        # Get consistent IDs for states
        consistent_state_id = self._get_consistent_state_id(raw_state_id, parse_table)
        consistent_stack = [self._get_consistent_state_id(s, parse_table) for s in raw_stack]
        self.log("Stack:", consistent_stack)
        self.log("State ID:", consistent_state_id)
        self.log("Raw State ID:", raw_state_id)
        self.log("Raw Stack:", raw_stack)
        return {
            'current_state': consistent_state_id,
            'stack': consistent_stack if top_k is None else consistent_stack[-top_k:],
            'value_stack': value_stack,
        }

    def parse_partial(self,  top_k=3):
        try:
            result = self.get_parser_state(self.interactive_parser, top_k=top_k)
            #result['remainder'] = self.current_remainder
            return result
        except Exception as e:
            return {
                'error': str(e),
                'success': False,
            }
        
    def advance_parser(self, sequence, top_k=3):
        self.feed_input(sequence)
        result = self.get_parser_state(self.interactive_parser, top_k=top_k)
        #result['remainder'] = self.current_remainder
        return result
    
    def _analyze_incremental(self, sequence):
        """
        Analyze a sequence incrementally, returning the parser state at each step
        """
        results = []
        
        try:
            # Get all tokens at once
            all_tokens, remainder = self.get_tokens_with_remainder(sequence)
            
            interactive = self.parser.parse_interactive('')
            
            # For each token:
            for i, token in enumerate(all_tokens):
                # Feed the next token
                interactive.feed_token(token)
                current_set = self.get_parser_state(interactive, top_k=None)
                current_set['text'] = all_tokens[:i + 1]
                results.append(current_set)
            results[-1]['remainder'] = remainder
                    
        except Exception as e:
            # If we fail, record the error
            results.append({
                'position': len(sequence),
                'error': str(e),
                'success': False,
            })
                
        return results

    def analyze_incremental_char(self, sequence):
        results = []
        
        # Keep track of how the stack evolves
        state_evolution = []
        
        for i in range(1, len(sequence) + 1):
            partial = sequence[:i]
            result = self.parse_partial(partial)
            
            if result['success']:
                state_evolution.append({
                    'text': partial,
                    'stack': result['stack'].copy() if 'stack' in result else []
                })
                
            results.append({
                'position': i,
                'text': partial,
                **result
            })
        
        # Store evolution for analysis
        self.state_evolution = state_evolution
        return results
    
    def reset(self):
        """
        Reset the parser state
        """
        self.tokens = []
        self.current_string = ""
        self.current_remainder = ""
        self.interactive_parser = self.parser.parse_interactive('')

    def validate_state_consistency(self):
        """Test that same input produces same states"""
        test_input = "simple test"
        state1 = self.advance_parser(test_input)
        self.reset()
        state2 = self.advance_parser(test_input)
        assert state1 == state2, "States not consistent!"

def extract_parser_states(grammar, sequence, top_k=None):
    return extractor.parse_partial(sequence, top_k)

def analyze_sequence(grammar, sequence):
    extractor = ParserStateExtractor(grammar)
    return extractor.analyze_incremental(sequence)

# Example usage
if __name__ == "__main__":


    json_grammar = r"""
        start: value

        ?value: object
            | array
            | string
            | SIGNED_NUMBER      -> number
            | "true"             -> true
            | "false"            -> false
            | "null"             -> null

        array  : "[" [value ("," value)*] "]"
        object : "{" [pair ("," pair)*] "}"
        pair   : string ":" value

        string : ESCAPED_STRING

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS
        %ignore WS
        """
    
    # Test with a partial expression
    sequence1 = '{"name": "Alexander", "active": tr'
    sequence2 = 'ue, "age": 25}'
    full_sequence = '{"name": "Alexander", "active": true, "age": 25}'
    full_sequence = '{"name": "Alice40", "age": 64, "active": false, "email": "user40@example.com", "tags": ["premium"], "preferences": {"notifications": "all", "theme": "dark"}\}'

    grammar = json_grammar
    extractor = ParserStateExtractor(grammar)
    """results = extractor.advance_parser(sequence1)
    print(results)
    results2 = extractor.advance_parser(sequence2)
    print(results2)
    
    extractor.reset()
    print("run two")
    extractor.feed_input(sequence1)
    results = extractor.parse_partial()
    print(results)
    extractor.feed_input(sequence2)
    results2 =extractor.parse_partial()
    print(results2)"""

    #extractor.reset()
    extractor.advance_parser(full_sequence)

    
    
    
    """print("\nIncremental analysis:")
    results = extractor._analyze_incremental(full_sequence)
    
    for r in results:
        print(f"('{r['text']}'): ")
        print(f"  State: {r['current_state']}")
        print(f"  Stack: {r['stack']}")"""

    """sequence = '{"name": "Alexander", "age": 25, "active": truxx'
    results = extractor.analyze_incremental(sequence)
    
    for r in results:
        if r['success']:
            print(f"Position {r['position']} ('{r['text']}'): ")
            print(f"  State: {r['current_state']}")
            print(f"  Stack: {r['stack']}")
        else:
            print(f"Position {r['position']} ('{r['text']}'): Error: {r['error']}")"""