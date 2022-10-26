import uuid
import sys

var = uuid.uuid4()


def print_func(continent: str = 'Asia') -> None:
    print('The name of continent is : ', continent)


def get_var() -> str:
    argv_safe = ''.join(ch for ch in sys.argv[0] if ch.isalnum())
    print(f"{argv_safe}")
    return var
