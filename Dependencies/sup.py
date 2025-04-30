from Dependencies.errorhandling import error_handling

# Check if conTrue == 1
@error_handling
def print_lock(conTrue: int):
    if conTrue == 1:
        return True
    elif conTrue == 0:
        return False

# Print debug if conTrue set to 1
@error_handling
def print_if(debug: str, conTrue: int):
    if print_lock(conTrue):
        print(debug)