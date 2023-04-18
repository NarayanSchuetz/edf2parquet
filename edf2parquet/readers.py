import json
import warnings
from typing import List, Dict, Any

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pytz

from edf2parquet.util import string_to_python_data_types


class ParquetEdfReader:

    def __init__(self, parquet_file_path: str):
        self._parquet_file_path = parquet_file_path
        self._pa_table = None

    @property
    def pa_table(self):
        if self._pa_table is None:
            self._pa_table = pq.read_table(self._parquet_file_path)
        return self._pa_table

    def get_pandas_dataframe(self, set_timezone: False) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame.

        Args:
            set_timezone: if True, the timezone of the recording will be set to the local timezone set in the metadata.

        Returns:
            A Pandas DataFrame with the signal data.
        """
        if set_timezone:
            file_metadata = self.get_file_header()
            if file_metadata["tz_recording"] == "":
                warnings.warn("No local recording timezone is set. The timezone of the recording will not be set.")
                return self.pa_table.to_pandas()
            df = self.pa_table.to_pandas()
            df.index = df.index.tz_localize(pytz.utc).tz_convert(pytz.timezone(file_metadata["tz_recording"]))
            return df
        return self.pa_table.to_pandas()

    def get_pyarrow_table(self) -> pa.Table:
        """
        Reads the parquet file and returns a PyArrow Table.

        Returns:
            A PyArrow Table with the signal data.
        """
        return self.pa_table

    def get_signal_labels(self) -> List[str]:
        """
        Returns a list of signal labels.

        Returns:
            A list of signal labels.
        """
        return [name for name in self.pa_table.schema.names if name != '__index_level_0__']

    def get_signal_header(self, signal_label: str) -> Dict[str, Any]:
        """
        Returns the signal header for the given signal label.

        Args:
            signal_label: the signal label.

        Returns:
            A dictionary of the signal header.
        """
        metadata = json.loads(self.pa_table.schema.field(signal_label).metadata[b'edf_signal_header'])
        metadata = string_to_python_data_types(metadata)
        return metadata

    def get_signal_headers(self) -> List[dict]:
        """
        Returns a list of signal headers.

        Returns:
            A list of dictionaries of the signal headers.
        """
        return [self.get_signal_header(signal_label) for signal_label in self.get_signal_labels()]

    def get_file_header(self) -> Dict[str, Any]:
        """
        Returns the file header.

        Returns:
            A dictionary of the file header.
        """
        metadata = json.loads(self.pa_table.schema.metadata[b'edf_file_header'])
        metadata["startdate"] = pd.to_datetime(metadata["startdate"])\
            .tz_localize(pytz.timezone(metadata["tz_startdatetime"]))
        return metadata
