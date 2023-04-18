import os.path

import pandas as pd
import pyarrow as pa
import pyedflib
import pyarrow.parquet as pq

import json
import os.path
from typing import Dict, Optional


class EdfToParquetConverter:

    def __init__(
            self,
            edf_file_path: str,
            datetime_index=True,
            default_signal_dtype=pa.float32(),
            parquet_output_dir: Optional[str] = None) -> None:
        """
        Args:
            edf_file_path: the absolute path to the EDF/EDF+ file.
            datetime_index: if True, the DatetimeIndex is inferred from the EDF startdate and sampling frequency,
                otherwise no index will be set.
            default_signal_dtype: the data type to use for the signal in the parquet file(s).
            parquet_output_dir: Optional. If specified, the parquet file(s) are stored in the specified directory.
        """
        self._datetime_index = datetime_index
        self._default_signal_dtype = default_signal_dtype
        self._edf_file = pyedflib.EdfReader(edf_file_path)
        self._parquet_output_dir = parquet_output_dir

    def __del__(self) -> None:
        self._edf_file.close()

    def __repr__(self) -> str:
        return f"EdfToParquetConverter({self._edf_file.getHeader()})"

    def convert(self) -> Optional[Dict[str, pa.Table]]:
        """
        Converts an EDF/EDF+ file to Apache Parquet file format.
        Each signal is stored in a separate parquet file with corresponding file and signal metadata stored as
        parquet file metadata.
        Furthermore, all signals are stored as a time-series with DatetimeIndex (inferred from the EDF startdate and
        sampling frequency).

        Args:

        Returns:
            None if parquet_output_dir is specified (parquet files are written to this directory, otherwise a dictionary
             with the signal labels as keys and the corresponding Apache Arrow Tables as values.
        """
        edf_file = self._edf_file
        parquet_output_dir = self._parquet_output_dir

        try:
            tables = self._convert_edf_to_pyarrow_tables()
            if isinstance(parquet_output_dir, str):
                os.makedirs(parquet_output_dir, exist_ok=True)
                for signal_label, table in tables.items():
                    pq.write_table(table, os.path.join(parquet_output_dir, f"{signal_label}.parquet"))
            else:
                return tables

        except Exception as e:
            raise e

        finally:
            edf_file.close()

    def _convert_edf_to_pyarrow_tables(self) -> Dict[str, pa.Table]:
        edf_file = self._edf_file
        n_signals = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()

        dfs = self._extract_edf()
        file_header = {str(k): str(v) for k, v in edf_file.getHeader().items()}
        file_header_metadata = {
            "edf_file_header".encode(): json.dumps(file_header).encode()
        }

        tables = {}
        for i in range(n_signals):
            df = dfs[signal_labels[i]]

            schema = pa.Schema.from_pandas(df, preserve_index=self._datetime_index)
            metadata = schema.metadata
            metadata = {**metadata, **file_header_metadata}

            signal_metadata = {
                "edf_signal_header".encode(): json.dumps({str(k): str(v) for k, v in
                                                          edf_file.getSignalHeader(i).items()}).encode()
            }
            schema = schema.add_metadata(metadata)
            field_names = schema.names
            existing_fields = [schema.field(field_name) for field_name in field_names]
            new_fields = [
                pa.field(existing_fields[0].name,
                         self._default_signal_dtype,
                         nullable=existing_fields[0].nullable,
                         metadata=signal_metadata),
                *existing_fields[1:]
            ]
            schema = pa.schema(new_fields, metadata=schema.metadata)

            table = pa.Table.from_pandas(df=df, schema=schema, preserve_index=self._datetime_index)
            tables[signal_labels[i]] = table
            dfs.pop(signal_labels[i])

        return tables

    def _extract_edf(self) -> Dict[str, pd.DataFrame]:
        edf_file = self._edf_file
        num_signals = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()

        data = {}
        for i in range(num_signals):
            sampling_rate = edf_file.getSampleFrequency(i)
            signal_data = edf_file.readSignal(i)
            n_samples = len(signal_data)

            freq_str = '{}N'.format(int(1e9 / sampling_rate))

            if self._datetime_index:
                time_points = pd.date_range(
                    start=pd.Timestamp(edf_file.getStartdatetime()),
                    periods=n_samples,
                    freq=freq_str,
                    inclusive='left'
                )
                df = pd.DataFrame(signal_data, index=time_points, columns=[signal_labels[i]])
                df.index = df.index.astype("datetime64[us]")  # cannot store nanoseconds in parquet
                data[signal_labels[i]] = df
            else:
                data[signal_labels[i]] = pd.DataFrame(signal_data, columns=[signal_labels[i]])
        return data