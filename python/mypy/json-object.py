from typing import Dict, Union
import datetime as dt

def print_json_lax(my_json: Dict[str, object]) -> None:
    print(str(my_json))

def print_json_strict(my_json: Dict[str, Union[str, float, dt.datetime]]) -> None:
    print(str(my_json))

my_lax_json = {
    'Hello': 'world',
    'Pi': 3.1415,
    'fibonacci': [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
    'gender': {
        'male': 4,
        'female': 5
    }
}

# Declaration
my_predictable_json: Dict[str, Union[str, float, dt.datetime]]

# Definition
my_predictable_json = {
    'ric': 'HSBC',
    'price': 3.14,
    'timestamp': dt.datetime(1970, 1, 1, 0, 0, 0)
}

print_json_lax(my_lax_json)
print_json_strict(my_predictable_json)