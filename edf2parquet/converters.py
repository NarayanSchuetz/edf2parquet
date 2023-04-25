import datetime
import os.path

import pandas as pd
import pyarrow as pa
import pyedflib
import pyarrow.parquet as pq

import json
import os.path
from typing import Dict, Optional, Union, Tuple, List

import pytz

from edf2parquet.readers import EdfReader
from edf2parquet.util import detect_nonuse_intervals


class EdfToParquetConverter:

    def __init__(
            self,
            edf_file_path: str,
            datetime_index=True,
            default_signal_dtype=pa.float32(),
            parquet_output_dir: Optional[str] = None,
            compression_codec: Union[str, dict] = "NONE",
            local_timezone: Optional[Tuple[pytz.BaseTzInfo, pytz.BaseTzInfo]] = None) -> None:
        """
        Args:
            edf_file_path: the absolute path to the EDF/EDF+ file.

            datetime_index: if True, the DatetimeIndex is inferred from the EDF startdate and sampling frequency,
                otherwise no index will be set.

            default_signal_dtype: the data type to use for the signal in the parquet file(s).

            parquet_output_dir: Optional. If specified, the parquet file(s) are stored in the specified directory.

            compression_codec: Optional. If specified, the parquet file(s) are compressed using the specified codec.
                Valid values are {‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’}, or a dict specifying a
                compression codec on a per-column basis (see PyArrow docs).

            local_timezone: Optional. A tuple of pytz timezones. The first entry identifies the physical timezone
                the EDF file was recorded in, the second entry identifies the timezone the EDF file startdate is in.
                E.g. (pytz.timezone('Europe/Berlin'), pytz.timezone('UTC')) for a file recorded in Berlin, but where
                the startdate has already been converted to UTC. If local_timezone is specified and datetime_index is
                True, the DatetimeIndex is converted to UTC before being stored in the parquet file(s), while the
                recording timezone is present in the metadata.
        """
        self._datetime_index = datetime_index
        self._default_signal_dtype = default_signal_dtype
        self._edf_file = pyedflib.EdfReader(edf_file_path)
        self._parquet_output_dir = parquet_output_dir
        self._compression_codec = compression_codec
        self._local_timezone = local_timezone

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
                    pq.write_table(table,
                                   os.path.join(parquet_output_dir, f"{signal_label}.parquet"),
                                   compression=self._compression_codec)
            else:
                return tables

        finally:
            edf_file.close()

    def _convert_edf_to_pyarrow_tables(self) -> Dict[str, pa.Table]:
        edf_file = self._edf_file
        n_signals = edf_file.signals_in_file
        signal_labels = edf_file.getSignalLabels()

        dfs = self._extract_edf()
        file_header = {str(k): str(v) for k, v in edf_file.getHeader().items()}

        if self._local_timezone is not None:
            file_header["tz_recording"] = str(self._local_timezone[0])
            file_header["tz_startdatetime"] = str(self._local_timezone[1])
        else:
            file_header["tz_recording"] = ""
            file_header["tz_startdatetime"] = ""

        file_header_metadata = {"edf_file_header".encode(): json.dumps(file_header).encode()}

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
            schema = schema.with_metadata(metadata)
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
                if self._local_timezone is not None:
                    start_datetime = pd.Timestamp(edf_file.getStartdatetime())\
                        .tz_localize(self._local_timezone[1])\
                        .tz_convert(pytz.UTC)\
                        .tz_localize(None)  # remove tzinfo, tends to be safer
                else:
                    start_datetime = pd.Timestamp(edf_file.getStartdatetime())

                time_points = pd.date_range(
                    start=start_datetime,
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


class AdvancedEdfToParquetConverter(EdfToParquetConverter):
    """
    Converts an EDF/EDF+ file to Apache Parquet file format.
    Each signal is stored in a separate parquet file with corresponding file and signal metadata stored as
    parquet file metadata.
    Furthermore, all signals are stored as a time-series with DatetimeIndex (inferred from the EDF startdate and
    sampling frequency).
    """

    def __init__(
            self,
            edf_file_path: str,
            parquet_output_dir: str,
            datetime_index=True,
            default_signal_dtype=pa.float32(),
            compression_codec: Union[str, dict] = "NONE",
            local_timezone: Optional[Tuple[pytz.BaseTzInfo, pytz.BaseTzInfo]] = None,
            group_by_sampling_freq: bool = False,
            exclude_signals: Optional[List[str]] = None,
            split_non_use_by_col: Optional[str] = None,
            split_non_use_max_std: Optional[float] = 0.1,
            split_non_use_min_duration_s: Optional[int] = 3600,
            **split_kwargs) -> None:
        """
        Args:
            edf_file_path: the absolute path to the EDF/EDF+ file.

            datetime_index: if True, the DatetimeIndex is inferred from the EDF startdate and sampling frequency,
                otherwise no index will be set.

            default_signal_dtype: the data type to use for the signal in the parquet file(s).

            parquet_output_dir: Optional. If specified, the parquet file(s) are stored in the specified directory.

            compression_codec: Optional. If specified, the parquet file(s) are compressed using the specified codec.
                Valid values are {‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’}, or a dict specifying a
                compression codec on a per-column basis (see PyArrow docs).

            local_timezone: Optional. A tuple of pytz timezones. The first entry identifies the physical timezone
                the EDF file was recorded in, the second entry identifies the timezone the EDF file startdate is in.
                E.g. (pytz.timezone('Europe/Berlin'), pytz.timezone('UTC')) for a file recorded in Berlin, but where
                the startdate has already been converted to UTC. If local_timezone is specified and datetime_index is
                True, the DatetimeIndex is converted to UTC before being stored in the parquet file(s), while the
                recording timezone is present in the metadata.

            group_by_sampling_freq: Optional. If True, signals with the same sampling frequency are grouped together
                and stored in the same parquet file. This can be useful if you want to reduce file number/ overhead.

            exclude_signals: Optional. Can be used to exclude signals from being converted. Defaults to None meaning
                all signals are converted.

            split_non_use_by_col: Optional. If set the column will be used to split the file into multiple parquet files
                based on non-use intervals (thus longer period where the signal std did not change markedly).
                Can for instance be useful to split an EDF multi-night recording into multiple parquet files each
                representing a night.

            split_non_use_max_std: Optional. Marks the maximum standard deviation of the signal that is considered
                non-use.

            split_non_use_min_duration_s: Optional. Marks the minimum duration of a non-use interval that is considered
                non-use.

            **split_kwargs: Optional. Additional keyword arguments passed to the split function.

        """
        super().__init__(edf_file_path, datetime_index, default_signal_dtype, parquet_output_dir,
                         compression_codec, local_timezone)
        self._edf_reader = EdfReader(edf_file_path)
        self._group_by_sampling_freq = group_by_sampling_freq
        self._exclude_signals = exclude_signals
        self._split_non_use_by_col = split_non_use_by_col
        self._split_non_use_max_std = split_non_use_max_std
        self._split_non_use_min_duration_s = split_non_use_min_duration_s
        self._split_kwargs = split_kwargs

    def convert(self) -> None:
        if self._split_non_use_by_col is not None:
            if not self._datetime_index:
                raise ValueError("Cannot split non-use intervals if datetime_index is False.")
            non_use_intervals = self._get_non_use_intervals()

        if self._group_by_sampling_freq:
            signal_dict = self._group_signal_labels_by_sampling_freq()
            for sf, labels in signal_dict.items():
                df = self._edf_reader.get_pandas_dataframe(*labels, set_timezone=False)
                if self._split_non_use_by_col is not None:
                    self._split_and_write_to_parquet(df, sf, non_use_intervals)
                else:
                    table = self._create_arrow_table(df)
                    filename = '{}_{}_{}.parquet'.format(self._edf_reader.edf_file.getPatientCode(),
                                                         self._edf_reader.edf_file.getStartdatetime().date(),
                                                         sf)
                    self._write_to_parquet({filename: table})

        else:
            for signal_label in self._edf_reader.edf_file.getSignalLabels():
                if self._exclude_signals is not None and signal_label in self._exclude_signals:
                    continue

                df = self._edf_reader.get_pandas_dataframe(signal_label, set_timezone=False)
                if self._split_non_use_by_col is not None:
                    self._split_and_write_to_parquet(df, signal_label, non_use_intervals)
                else:
                    table = self._create_arrow_table(df)
                    filename = '{}_{}_{}.parquet'.format(self._edf_reader.edf_file.getPatientCode(),
                                                         self._edf_reader.edf_file.getStartdatetime().date(),
                                                         signal_label)
                    self._write_to_parquet({filename: table})

    def _split_and_write_to_parquet(self,
                                    df: pd.DataFrame,
                                    label: str,
                                    non_use_intervals: List[Tuple[datetime.datetime, datetime.datetime]]) -> None:

        if len(non_use_intervals) == 0:
            return

        # Sort the non-use intervals just in case they are not sorted
        non_use_intervals = sorted(non_use_intervals, key=lambda x: x[0])

        # this is inconsistent with the above, we should extract use-times that would make everything more
        # straightforward
        for start, stop in non_use_intervals:
            # Slice the dataframe before the current non-use interval
            split_df = df.loc[df.index < start]
            filename = '{}_{}_{}_{}_{}.parquet'.format(self._edf_reader.edf_file.getPatientCode(),
                                                       self._edf_reader.edf_file.getStartdatetime().date(),
                                                       str(split_df.index[0].timestamp()),
                                                       str(split_df.index[-1].timestamp()),
                                                       label)
            table = self._create_arrow_table(split_df)
            self._write_to_parquet({filename: table})

            # Update the dataframe to include only data after the current non-use interval
            df = df.loc[df.index >= stop]

        if not df.empty:
            filename = '{}_{}_{}_{}_{}.parquet'.format(self._edf_reader.edf_file.getPatientCode(),
                                                       self._edf_reader.edf_file.getStartdatetime().date(),
                                                       str(df.index[0].timestamp()),
                                                       str(df.index[-1].timestamp()),
                                                       label)
            table = self._create_arrow_table(df)
            self._write_to_parquet({filename: table})

    def _get_non_use_intervals(self) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        df = self._edf_reader.get_pandas_dataframe(self._split_non_use_by_col, set_timezone=False)
        return detect_nonuse_intervals(df[self._split_non_use_by_col],
                                       std_threshold=self._split_non_use_max_std,
                                       min_interval_duration_seconds=self._split_non_use_min_duration_s,
                                       **self._split_kwargs)

    def _write_to_parquet(self, table_dict: Dict[str, pa.Table]) -> None:
        for filename, table in table_dict.items():
            filepath = os.path.join(self._parquet_output_dir, filename)
            os.makedirs(self._parquet_output_dir, exist_ok=True)
            pq.write_table(table, filepath, compression=self._compression_codec)

    def _create_arrow_table(self, df: pd.DataFrame) -> pa.Table:
        if self._datetime_index:
            df.index = df.index.astype("datetime64[us]")
        file_header_metadata = self._extract_edf_file_header_metadata()

        schema = pa.Schema.from_pandas(df, preserve_index=self._datetime_index)
        arrow_metadata = schema.metadata
        arrow_metadata = {**arrow_metadata, **file_header_metadata}
        schema = schema.with_metadata(arrow_metadata)

        field_names = schema.names
        existing_fields = [schema.field(field_name) for field_name in field_names]

        new_fields = []
        for i in range(len(existing_fields) - 1 if self._datetime_index else len(existing_fields)):
            field = pa.field(existing_fields[i].name,
                             self._default_signal_dtype,
                             nullable=existing_fields[i].nullable,
                             metadata=self._extract_signal_header_metadata(existing_fields[i].name))
            new_fields.append(field)

        if self._datetime_index:
            new_fields.append(existing_fields[-1])

        schema = pa.schema(new_fields, metadata=schema.metadata)
        return pa.Table.from_pandas(df=df, schema=schema, preserve_index=self._datetime_index)

    def _extract_edf_file_header_metadata(self) -> Dict[bytes, bytes]:
        file_header = {str(k): str(v) for k, v in self._edf_reader.edf_file.getHeader().items()}

        if self._local_timezone is not None:
            file_header["tz_recording"] = str(self._local_timezone[0])
            file_header["tz_startdatetime"] = str(self._local_timezone[1])
        else:
            file_header["tz_recording"] = ""
            file_header["tz_startdatetime"] = ""

        return {"edf_file_header".encode(): json.dumps(file_header).encode()}

    def _extract_signal_header_metadata(self, signal_label: str) -> Dict[bytes, bytes]:
        return {
                "edf_signal_header".encode(): json.dumps({str(k): str(v) for k, v in
                                                          self._edf_reader.edf_file.getSignalHeader(
                                                              self._edf_reader.signal_label_to_index(signal_label)
                                                          ).items()}).encode()
            }

    def _group_signal_labels_by_sampling_freq(self) -> Dict[str, List[str]]:
        f = self._edf_reader.edf_file
        n_signals = f.signals_in_file

        grouped_labels = {}
        for i in range(n_signals):
            label = f.getLabel(i)

            if self._exclude_signals is not None and label in self._exclude_signals:
                continue

            sampling_frequency = f.getSampleFrequency(i)
            sf_str = str(sampling_frequency)

            if sf_str in grouped_labels:
                grouped_labels[sf_str].append(label)
            else:
                grouped_labels[sf_str] = [label]

        return grouped_labels
