from typing import Union, Dict


def _try_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def _try_int(value):
    try:
        return int(value)
    except ValueError:
        return _try_float(value)


def string_to_python_data_types(input_str_dict: Dict[str, str]) -> Dict[str, Union[str, int, float]]:
    """
    Convert a dictionary of string values to their corresponding Python numeric data types if possible.

    Args:
        input_str_dict: a dictionary of string key and values.

    Returns:
        The same dictionary but with the values converted to python numeric data types where possible.
    """
    for key, value in input_str_dict.items():
        input_str_dict[key] = _try_int(value)

    return input_str_dict
