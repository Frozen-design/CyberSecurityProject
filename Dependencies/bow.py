import re
from collections import Counter
from Dependencies.errorhandling import error_handling

@error_handling
def bag_of_words(text, tokenizer=None):
    """Convert text to a bag-of-words representation."""
    if tokenizer is None:
        tokenizer = simple_tokenizer
    tokens = tokenizer(text)
    return Counter(tokens)

@error_handling
def vocabulary(texts, tokenizer=None, saved_vocab=None):
    """Create a vocabulary from a list of texts."""
    if tokenizer is None:
        tokenizer = simple_tokenizer

    vocab = set()
    if saved_vocab is not None:
        if isinstance(saved_vocab, list):
            vocab = set(saved_vocab)
        elif isinstance(saved_vocab, str):
            with open(saved_vocab, 'r', encoding='utf-8') as f:
                vocab = set(f.read().splitlines())
        elif isinstance(saved_vocab, set):
            vocab = saved_vocab

    for text in texts:
        tokens = tokenizer(text)
        vocab.update(tokens)
    return sorted(vocab)

@error_handling
def simple_tokenizer(text):
    """A simple tokenizer that splits text into words."""
    return [x.lower() for x in re.findall(r'\b\w+\b', text)]