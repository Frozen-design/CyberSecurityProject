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
def update_bow(bow, text, tokenizer=None):
    """Update a bag-of-words with new text."""
    if tokenizer is None:
        tokenizer = simple_tokenizer
    tokens = tokenizer(text)
    bow.update(tokens)
    return bow
    
@error_handling
def simple_tokenizer(text):
    """A simple tokenizer that splits text into words."""
    return [x.lower() for x in re.findall(r'\b\w+\b', text)]