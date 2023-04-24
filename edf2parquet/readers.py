import json
import warnings
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyedflib
import pytz

from edf2parquet.util import string_to_python_data_types


class ParquetReader:

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


class EdfReader:

    def __init__(self,
                 edf_file_path: str,
                 patient_id: Optional[str] = None,
                 datetime_index: bool = True,
                 local_timezone: Optional[Tuple[pytz.BaseTzInfo, pytz.BaseTzInfo]] = None):
        """
        Args:
            edf_file_path: the absolute path to the .EDF file.

            patient_id: if set, will update the patient_id header field in the .EDF file to the given value. Value must
                be in the format "X X X X", e.g. "2 M 20-DEC-1972 DomoSMA" -> "{PATIENT_ID} {GENDER} {DOB} {NAME}.

                See https://www.edfplus.info/specs/edfplus.html#edfplusandedf:
                The 'local patient identification' field must start with the subfields (subfields do not contain,
                but are separated by, spaces):
                - the code by which the patient is known in the hospital administration.
                - sex (English, so F or M).
                - birthdate in dd-MMM-yyyy format using the English 3-character abbreviations of the month in capitals.
                  02-AUG-1951 is OK, while 2-AUG-1951 is not.
                - the patients name.

            datetime_index: if True, the index of the returned DataFrame will be a DatetimeIndex. If False, no index
                will be set.

            local_timezone: local_timezone: Optional. A tuple of pytz timezones. The first entry identifies the physical
                timezone the EDF file was recorded in, the second entry identifies the timezone the EDF file startdate is
                in. E.g. (pytz.timezone('Europe/Berlin'), pytz.timezone('UTC')) for a file recorded in Berlin, but where
                the startdate has already been converted to UTC. If local_timezone is specified and datetime_index is
                True, the DatetimeIndex is converted to UTC before being stored in the parquet file(s), while the
                recording timezone is present in the metadata.
        """
        self._edf_file_path = edf_file_path
        self._patient_id = patient_id
        self._edf_file = None
        self._datetime_index = datetime_index
        self._local_timezone = local_timezone

    def __repr__(self):
        return f"EdfReader({self._edf_file_path})"

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._edf_file is not None:
            self._edf_file.close()

    def __del__(self):
        if self._edf_file is not None:
            self._edf_file.close()

    @property
    def edf_file(self) -> pyedflib.EdfReader:
        if self._edf_file is None:
            self._read_edf_file()
        return self._edf_file

    def _read_edf_file(self):
        """
        Attempts to read the specified .EDF file. If the file cannot be read, attempts to correct the header by
        replacing non ascii printable characters in the signal header with '.' and updating the patient_id field to the
        given value (if set). If the file cannot be read after correcting the header, raises an OSError.
        In our experience, these are the two most common reasons for corrupted .EDF files but of course not the only
        ones.

        Raises:
            OSError: if the file cannot be read after correcting the header.
        """
        try:
            self._edf_file = pyedflib.EdfReader(self._edf_file_path)
        except OSError as e:
            if not self._edf_file_path.endswith(".edf"):
                raise OSError(f"Unable to read file: {e}. file suffix suggests it is not an .edf file.")

            warnings.warn(f"Unable to read .EDF file: {e}. Attempting to correct the header.")
            self._correct_edf_header()
            self._edf_file = pyedflib.EdfReader(self._edf_file_path)

    def _signal_label_to_index(self, signal_label: str) -> int:
        """
        Returns the signal index for the given signal label.

        Args:
            signal_label: the signal label.

        Returns:
            The signal index.
        """
        return self.edf_file.getSignalLabels().index(signal_label)

    def get_pandas_dataframe(self, *signal_labels: str, set_timezone: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the signal data for the given signal label(s).
        NOTE: all provided signals must have the same sampling rate.

        Args:
            signal_labels: the signal label(s).
            set_timezone: if True, the DatetimeIndex is converted to the local timezone before being returned.
                Otherwise, the DatetimeIndex is in UTC but not TZ-aware.
                This will only have an effect if datetime_index is True and local_timezone is set.

        Returns:
            A DataFrame with the signal data.

        Raises:
            ValueError: if the sampling rate of the provided signals is not the same.
        """
        if len(signal_labels) == 0:
            return pd.DataFrame()

        edf_file = self.edf_file
        data = {}
        sampling_rate = edf_file.getSampleFrequency(self._signal_label_to_index(signal_labels[0]))
        for signal_label in signal_labels:
            i = self._signal_label_to_index(signal_label)
            if sampling_rate != edf_file.getSampleFrequency(i):
                raise ValueError(f"Signal {signal_label} has a different sampling rate than the other signals.")
            data[signal_label] = edf_file.readSignal(i)

        n_samples = len(data[signal_labels[0]])
        freq_str = '{}N'.format(int(1e9 / sampling_rate))

        if self._datetime_index:
            if self._local_timezone is not None:
                if set_timezone:
                    start_datetime = pd.Timestamp(edf_file.getStartdatetime()) \
                        .tz_localize(self._local_timezone[1]) \
                        .tz_convert(self._local_timezone[0])
                else:
                    start_datetime = pd.Timestamp(edf_file.getStartdatetime()) \
                        .tz_localize(self._local_timezone[1]) \
                        .tz_convert(pytz.UTC) \
                        .tz_localize(None)  # remove tzinfo, tends to be safer

            else:
                start_datetime = pd.Timestamp(edf_file.getStartdatetime())

            time_points = pd.date_range(
                start=start_datetime,
                periods=n_samples,
                freq=freq_str,
                inclusive='left'
            )
            df = pd.DataFrame(data, index=time_points)
            #df.index = df.index.astype("datetime64[us]")  # cannot store nanoseconds in parquet
            return df
        else:
            return pd.DataFrame(data)

    def _correct_edf_header(self) -> None:
        """
        Corrects the header of the .EDF file by updating the patient_id field to the given value and replacing non
        ascii printable characters in the signal header with '.'.
        Returns:

        """
        with open(self._edf_file_path, 'rb') as f:
            header_data = bytearray()
            f.seek(0)

            header_data.extend(f.read(8))  # Read the version field
            if self._patient_id is not None and len(self._patient_id.split(" ")) == 3:
                assert len(self._patient_id) <= 80, "Patient ID must be less than 80 characters."
                header_data.extend(self._patient_id.encode('ascii').ljust(80))
                f.read(80)
            elif self._patient_id is not None and len(self._patient_id.split(" ")) != 3:
                warnings.warn("Provided patient_id header format is invalid, setting default id: 'X X X X'.")
                header_data.extend("X X X X".encode('ascii').ljust(80))
            else:
                header_data.extend(f.read(80))

            header_data.extend(f.read(80))  # Read the record_id field
            header_data.extend(f.read(8))  # Read the start_date field
            header_data.extend(f.read(8))  # Read the start_time field
            header_data.extend(f.read(8))  # Read the num_header_bytes field
            header_data.extend(f.read(44))  # Read the reserved field
            header_data.extend(f.read(8))  # Read the num_data_records field
            header_data.extend(f.read(8))  # Read the data_record_duration field

            ns = f.read(4).decode("ascii")
            num_signals = int(ns)
            header_data.extend(ns.encode('ascii'))

            signal_headers_data = bytearray()

            for field, nbytes, field_type in [
                ('label', 16, str),
                ('transducer_type', 80, str),
                ('physical_dimension', 8, str),
                ('physical_min', 8, float),
                ('physical_max', 8, float),
                ('digital_min', 8, int),
                ('digital_max', 8, int),
                ('prefiltering', 80, str),
                ('num_samples', 8, int),
                ('reserved', 32, str)
            ]:
                for _ in range(num_signals):
                    field_data = f.read(nbytes)
                    try:
                        decoded_field = field_data.decode("ascii")
                        encoded_field = decoded_field.encode('ascii')
                        encoded_field += b' ' * (nbytes - len(encoded_field))
                    except UnicodeDecodeError:
                        encoded_field = b' ' * nbytes
                    signal_headers_data.extend(encoded_field)

        # Write the modified header and signal headers back to the file
        with open(self._edf_file_path, 'r+b') as f:
            f.seek(0)
            f.write(header_data)
            f.write(signal_headers_data)
