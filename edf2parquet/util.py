from typing import Dict, Any, List, Tuple

import pandas as pd


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


def string_to_python_data_types(input_str_dict: Dict[str, str]) -> Dict[str, Any]:
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


def detect_nonuse_intervals(
        series: pd.Series,
        pd_freq="1s",
        std_threshold=0.1,
        min_interval_duration_seconds=3600) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Efficient way to detect intervals between which a sensor is not used if the resampled standard deviation falls below
    a given threshold.

    Args:
        series: the pandas series representing a sensor's signal

        pd_freq: the pandas frequency to which to resample the signal (e.g. "1min", "30s", "1d")

        std_threshold: the threshold below which the signals resampled standard deviation is considered to be associated
         with non-use

         min_interval_duration_seconds: the minimum duration of a non-use interval to be considered as such.

    Returns:
        a list of tuples, each tuple representing a start and end time of a non-use interval.
    """
    assert isinstance(series.index, pd.DatetimeIndex), "pandas series must possess a datetime index."
    series = series.copy()
    is_unused_series = series.resample(pd_freq, origin=series.index[0]).std()
    is_unused_series = is_unused_series < std_threshold

    starts = is_unused_series[(is_unused_series) & (~is_unused_series.shift(1).fillna(False))].dropna().index
    ends = is_unused_series[(is_unused_series) & (~is_unused_series.shift(-1).fillna(False))].dropna().index
    return [(start, end) for start, end in zip(starts, ends) if
            (end - start).total_seconds() > min_interval_duration_seconds]
