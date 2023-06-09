import glob
import json
import os
import shutil

import pandas as pd
import pyarrow.parquet as pq
import pytz
import requests


from edf2parquet.converters import EdfToParquetConverter, AdvancedEdfToParquetConverter


class BaseConverterTest:

    _URL = 'http://paulbourke.net/dataformats/edf/test.edf'
    _FILE_NAME = 'test_resources/temp_test.edf'

    def setup_method(self, test_method):
        file_name = self._FILE_NAME
        url = self._URL

        if os.path.isfile(file_name):
            return

        response = requests.get(url)

        if response.status_code == 200:
            with open(file_name, 'wb') as file:
                file.write(response.content)
        else:
            print("Error: Unable to download the file skip test or add a new valid file url. "
                  "Status code:", response.status_code)

    # def teardown_method(self, test_method):
    #     if os.path.isfile(self._FILE_NAME):
    #         os.remove(self._FILE_NAME)


class Test_Integration_EdfToParquetConverter(BaseConverterTest):

    def test_integration_convert_edf_to_pyarrow_tables(self):
        converter = EdfToParquetConverter(self._FILE_NAME, datetime_index=True)
        result = converter._convert_edf_to_pyarrow_tables()
        assert len(result) == 20
        signal_metadata = json.loads(result["EEG A2-A1"].schema.field("EEG A2-A1").metadata[b'edf_signal_header'])
        assert signal_metadata['label'] == 'EEG A2-A1'
        file_metadata = json.loads(result["EEG A2-A1"].schema.metadata[b'edf_file_header'])
        assert file_metadata['birthdate'] == 'F'
        assert isinstance(result['EEG A2-A1'].to_pandas().index, pd.DatetimeIndex)

    def test_integration_convert_edf_to_pyarrow_tables_noindex(self):
        converter = EdfToParquetConverter(self._FILE_NAME, datetime_index=False)
        result = converter._convert_edf_to_pyarrow_tables()
        assert not isinstance(result['EEG A2-A1'].to_pandas().index, pd.DatetimeIndex)

    def test_integration_end2end_without_outputdir(self):
        converter = EdfToParquetConverter(self._FILE_NAME, datetime_index=True)
        result = converter.convert()
        assert len(result) == 20
        signal_metadata = json.loads(result["EEG A2-A1"].schema.field("EEG A2-A1").metadata[b'edf_signal_header'])
        assert signal_metadata['label'] == 'EEG A2-A1'
        file_metadata = json.loads(result["EEG A2-A1"].schema.metadata[b'edf_file_header'])
        assert file_metadata['birthdate'] == 'F'
        assert isinstance(result['EEG A2-A1'].to_pandas().index, pd.DatetimeIndex)

    def test_integration_end2end_with_outputdir(self):
        output_dir = "temp_output_dir"
        converter = EdfToParquetConverter(self._FILE_NAME,
                                          datetime_index=True,
                                          parquet_output_dir=output_dir,
                                          compression_codec="GZIP",
                                          local_timezone=(pytz.timezone("Europe/Paris"), pytz.timezone("Europe/Paris")))
        converter.convert()

        # Check if the output directory was created
        assert os.path.isdir(output_dir)

        # Check if files were actually created in the output directory
        parquet_files = glob.glob(os.path.join(output_dir, "*.parquet"))
        assert len(parquet_files) == 20

        # Read one of the Parquet files and test its content
        example_parquet_file = parquet_files[0]
        table = pq.read_table(example_parquet_file)
        signal_label = os.path.splitext(os.path.basename(example_parquet_file))[0]

        signal_metadata = json.loads(table.schema.field(signal_label).metadata[b'edf_signal_header'])
        assert signal_metadata['label'] == signal_label

        file_metadata = json.loads(table.schema.metadata[b'edf_file_header'])
        assert file_metadata['birthdate'] == 'F'
        assert file_metadata['tz_recording'] == 'Europe/Paris'
        assert file_metadata['tz_startdatetime'] == 'Europe/Paris'

        assert isinstance(table.to_pandas().index, pd.DatetimeIndex)

        # Clean up the output directory
        shutil.rmtree(output_dir)


class Test_Integration_AdvancedEdfToParquetConverter(BaseConverterTest):

    def test_integration_end2end_bundled(self):
        output_dir = "temp_output_dir"
        converter = AdvancedEdfToParquetConverter(self._FILE_NAME,
                                                  group_by_sampling_freq=True,
                                                  parquet_output_dir=output_dir,
                                                  datetime_index=True,
                                                  compression_codec="GZIP",
                                                  local_timezone=(pytz.timezone("Europe/Paris"),
                                                                  pytz.timezone("Europe/Paris")))
        converter.convert()

        # Check if the output directory was created
        assert os.path.isdir(output_dir)

        # Check if files were actually created in the output directory
        parquet_files = glob.glob(os.path.join(output_dir, "*.parquet"))
        assert len(parquet_files) == 1

        # Read one of the Parquet files and test its content
        example_parquet_file = parquet_files[0]
        table = pq.read_table(example_parquet_file)
        signal_label = os.path.splitext(os.path.basename(example_parquet_file))[0]

        signal_metadata = json.loads(table.schema.field("EEG A2-A1").metadata[b'edf_signal_header'])
        assert signal_metadata['label'] == "EEG A2-A1"

        file_metadata = json.loads(table.schema.metadata[b'edf_file_header'])
        assert file_metadata['birthdate'] == 'F'
        assert file_metadata['tz_recording'] == 'Europe/Paris'
        assert file_metadata['tz_startdatetime'] == 'Europe/Paris'

        assert isinstance(table.to_pandas().index, pd.DatetimeIndex)

        # Clean up the output directory
        shutil.rmtree(output_dir)

    def test_integration_end2end_individual_files(self):
        output_dir = "temp_output_dir"
        converter = AdvancedEdfToParquetConverter(self._FILE_NAME,
                                                  group_by_sampling_freq=False,
                                                  parquet_output_dir=output_dir,
                                                  datetime_index=True,
                                                  compression_codec="GZIP",
                                                  local_timezone=(pytz.timezone("Europe/Paris"),
                                                                  pytz.timezone("Europe/Paris")))
        converter.convert()

        # Check if the output directory was created
        assert os.path.isdir(output_dir)

        # Check if files were actually created in the output directory
        parquet_files = glob.glob(os.path.join(output_dir, "*.parquet"))
        assert len(parquet_files) == 20

        # Read one of the Parquet files and test its content
        example_parquet_file = parquet_files[0]
        table = pq.read_table(example_parquet_file)
        signal_label = os.path.splitext(os.path.basename(example_parquet_file))[0]
        signal_label = signal_label.split("_")[-1]

        signal_metadata = json.loads(table.schema.field(signal_label).metadata[b'edf_signal_header'])
        assert signal_metadata['label'] == signal_label

        file_metadata = json.loads(table.schema.metadata[b'edf_file_header'])
        assert file_metadata['birthdate'] == 'F'
        assert file_metadata['tz_recording'] == 'Europe/Paris'
        assert file_metadata['tz_startdatetime'] == 'Europe/Paris'

        assert isinstance(table.to_pandas().index, pd.DatetimeIndex)

        # Clean up the output directory
        shutil.rmtree(output_dir)
