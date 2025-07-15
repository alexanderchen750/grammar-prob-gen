from lark import Lark
from lark.lexer import Token

"""
Class for testing the simple GAD grammar: 00000 or any str starting w 1 and is 5 tokens of length.
"""
if __name__ == "__main__":
    with open("gad.lark", 'r') as f:
        grammar_text = f.read()

    parser = Lark(grammar_text, parser='lalr', lexer='contextual')

    test_examples = ["10001","00000", "10000", "11111", "111111", "01000", "00001"]

    for test_example in test_examples:
        interactive_parser = parser.parse_interactive('')
        try:
            stream = list(parser.lex(test_example))
            try:
                tree = parser.parse(test_example)
                child = tree.children
            except Exception as e:
                print(e)

            for token in stream:
                print(token)
                print(interactive_parser.accepts())
                if 'BIT' in interactive_parser.accepts() and token.value == '1':#shortcut
                    token = Token('BIT', token.value)
                    print(token)

                interactive_parser.feed_token(token)
            print(f"success: {test_example}", tree)
        except Exception as e:
            print(f"fail: {test_example}", e)

