import re
from collections import Counter
from typing import List
from Dependencies.errorhandling import error_handling
import scipy.sparse as sp
    
@error_handling
def simple_tokenizer(text: str) -> list:
    """A simple tokenizer that splits text into words."""
    return [str(x).lower() for x in re.findall(r'\w+', text)]

@error_handling
def sparse_vectorize(tokens, vocab: set):
    """Convert a list of tokens into a sparse vector based on the vocabulary."""
    vector = Counter(tokens)
    sparse_vector = sp.lil_matrix((1, len(vocab)), dtype=int)
    word_to_index = {word: i for i, word in enumerate(sorted(vocab))}
    for word, count in vector.items():
        if word in word_to_index:
            sparse_vector[0, word_to_index[word]] = count
    
    return sparse_vector

def vectorize_data(data: List[Counter], vocab: set) -> sp.lil_matrix:
    """Convert a Counter object into a sparse matrix based on the vocabulary."""
    sparse_matrix = sp.lil_matrix((len(data), len(vocab)), dtype=int)
    word_to_index = {word: i for i, word in enumerate(sorted(vocab))}
    
    for i, count in enumerate(data):
        for word, cnt in count.items():
            if word in word_to_index:
                sparse_matrix[i, word_to_index[word]] = cnt
            
    return sparse_matrix