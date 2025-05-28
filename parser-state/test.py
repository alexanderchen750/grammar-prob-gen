from lark import Lark
from lark.lexer import Token
from lark.exceptions import UnexpectedInput
from lark.tree import Tree
from lark.visitors import Transformer   
from lark.lexer import LexerState
from lark.lexer import TextSlice
from lark.lexer import BasicLexer


def create_lexer_for_grammar(grammar_text):
    """Create a standalone lexer for a grammar"""
    # First, create a parser to get access to its lexer configuration
    parser = Lark(grammar_text, parser='lalr')
    
    # Then extract the lexer configuration
    lexer_conf = parser.lexer_conf
    
    # Create a BasicLexer directly
    return BasicLexer(lexer_conf)


def lex_with_basic_lexer(lexer, text):
    """Lex text using a BasicLexer, handling errors gracefully"""
    # Create lexer state
    lex_state = LexerState(TextSlice(text, 0, len(text)))
    
    tokens = []
    pos = 0
    try:
        # Try to lex the entire input
        while pos < len(text):
            try:
                token = lexer.next_token(lex_state, None)
                tokens.append(token)
                pos = token.end_pos
            except EOFError:
                break
        
        remainder = ""
        success = True
    except Exception as e:
        # Lexing error - return what we've got so far
        remainder = text[pos:]
        success = False
    
    return {
        'tokens': tokens,
        'remainder': remainder,
        'success': success,
        'last_position': pos
    }

if __name__ == "__main__":
    # Example grammar
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
    
    # Create lexer
    lexer = create_lexer_for_grammar(json_grammar)
    
    # Example text to lex
    text = '{"name": "Alexander", "age": 25, "active": true'
    
    # Lex the text
    result = lex_with_basic_lexer(lexer, text)
    
    # Print the result
    print("Tokens:", result['tokens'])
    print("Remainder:", result['remainder'])
    print("Success:", result['success'])
    print("Last position:", result['last_position'])