import pandas as pd
import pyarrow as pa
import pytest
import pytz

from edf2parquet.readers import ParquetReader, EdfReader


class TestIntegration_ParquetEdfReader:

    @pytest.fixture
    def test_file_path(self):
        return "test_resources/EEG T5-LE.parquet"

    def test_get_pandas_dataframe(self, test_file_path):
        reader = ParquetReader(test_file_path)
        df = reader.get_pandas_dataframe(set_timezone=False)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.index.tzinfo is None

    def test_get_pandas_dataframe_with_timezone(self, test_file_path):
        reader = ParquetReader(test_file_path)
        df = reader.get_pandas_dataframe(set_timezone=True)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tzinfo == pytz.timezone("Europe/Paris")

    def test_get_pyarrow_table(self, test_file_path):
        reader = ParquetReader(test_file_path)
        table = reader.get_pyarrow_table()
        assert isinstance(table, pa.Table)

    def test_get_signal_labels(self, test_file_path):
        reader = ParquetReader(test_file_path)
        signal_labels = reader.get_signal_labels()
        assert isinstance(signal_labels, list)
        assert len(signal_labels) == 1
        assert signal_labels == ["EEG T5-LE"]

    def test_get_file_header(self, test_file_path):
        reader = ParquetReader(test_file_path)
        file_header = reader.get_file_header()
        assert isinstance(file_header, dict)
        assert len(file_header) == 12
        assert isinstance(file_header["startdate"], pd.Timestamp)
        assert file_header["tz_recording"] == "Europe/Paris"
        assert file_header["tz_startdatetime"] == "Europe/Paris"

    def test_get_signal_header(self, test_file_path):
        reader = ParquetReader(test_file_path)
        signal_header = reader.get_signal_header("EEG T5-LE")
        assert isinstance(signal_header, dict)
        assert len(signal_header) == 9
        assert isinstance(signal_header["digital_min"], int)
        assert isinstance(signal_header["physical_min"], float)
        assert signal_header["digital_min"] == -32768
        assert signal_header["physical_min"] == -3277.0
        assert signal_header["label"] == "EEG T5-LE"


class TestEdfReader:

    @pytest.fixture
    def file_path(self):
        return "test_resources/temp_test.edf"

    def test_get_pandas_dataframe(self, file_path):
        reader = EdfReader(file_path, datetime_index=True,
                           local_timezone=(pytz.timezone("Europe/Paris"),
                                           pytz.timezone("Europe/Paris")))
        df = reader.get_pandas_dataframe("EEG F3-LE", set_timezone=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tzinfo == pytz.timezone("Europe/Paris")

    def test_get_pandas_dataframe_no_datetime_index(self, file_path):
        reader = EdfReader(file_path, datetime_index=False)
        df = reader.get_pandas_dataframe("EEG F3-LE", set_timezone=False)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.RangeIndex)

    def test_get_pandas_dataframe_no_timezone_set(self, file_path):
        reader = EdfReader(file_path, datetime_index=True, local_timezone=(pytz.timezone("Europe/Paris"),
                                                                           pytz.timezone("Europe/Paris")))
        df = reader.get_pandas_dataframe("EEG F3-LE", set_timezone=False)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tzinfo is None

    def test_no_signals(self, file_path):
        reader = EdfReader(file_path, datetime_index=True, local_timezone=(pytz.timezone("Europe/Paris"),
                                                                           pytz.timezone("Europe/Paris")))
        df = reader.get_pandas_dataframe(set_timezone=False)
        assert df.empty

    def test_get_pandas_dataframe_multilabel(self, file_path):
        reader = EdfReader(file_path,
                           datetime_index=True,
                           local_timezone=(pytz.timezone("Europe/Paris"),
                                           pytz.timezone("Europe/Paris")))
        df = reader.get_pandas_dataframe("EEG T5-LE", "EEG F3-LE", set_timezone=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 2
        assert df["EEG T5-LE"].std() > 0
        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
