from lark import Lark
"""
Class for testing the simple GAD grammar: 00000 or any str starting w 1 and is 5 tokens of length.
"""
if __name__ == "__main__":
    with open("gad.lark", 'r') as f:
        grammar_text = f.read()

    parser = Lark(grammar_text, parser='lalr')
    test_examples = ["00000", "10000", "11111", "111111", "01000", "00001"]

    for test_example in test_examples:
        try:
            tree = parser.parse(test_example)
            print(f"success: {test_example}" ,tree)
        except Exception as e:
            print(f"fail: {test_example}",e)

