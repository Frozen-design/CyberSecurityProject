from Dependencies.errorhandling import error_handling
import pandas as pd
import numpy as np
# Check if conTrue == 1
@error_handling
def print_lock(conTrue: int):
    if conTrue == 1:
        return True
    elif conTrue == 0:
        return False

# Print debug if conTrue set to 1
@error_handling
def print_if(debug: str, conTrue = 0):
    if print_lock(conTrue):
        print(debug)

@error_handling
def read_file(file_path: str):
    """Read a file and return its content."""
    with open(file_path, 'r') as file:
        return file.read()

@error_handling
def parse_csv(file_path: str, headers: bool = True, names=['Text', 'Class'], dtype={'Text': str, 'Class': int}, delimiter = ",", quotechar = '"') -> pd.DataFrame:
    """Parse a CSV file and return its content as a list of dictionaries."""
    data = pd.read_csv(file_path, header=(0 if headers else None), names=names, dtype=dtype, na_filter=True, delimiter=delimiter, encoding='utf-8', quotechar=quotechar)
    if data.empty:
        raise ValueError("The CSV file is empty or not formatted correctly.")
    data = data.dropna(subset=['Text', 'Class'])
    return data

