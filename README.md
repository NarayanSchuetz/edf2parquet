# edf2parquet
Simple utility package to convert EDF/EDF+ files into Apache Parquet format 
while preserving the EDF file header information and signal headers metadata information and some nice enhanced features:
- handling of non-strictly EDF compliant .EDF headers (e.g. UTF-8 characters in the header, etc.).
- automatic conversion of the EDF file header start date and signal sampling frequency to a pd.DatetimeIndex with the correct timezone and frequency for easy Pandas interoperability (at the cost of slightly bigger file sizes of course).
- skipping of specific signals during conversion
- bundling signals with the same sampling frequency into a single parquet file
- splitting of EDF files by non-use periods (e.g. if a file consists of continuous multiple nights, and you want to split it into a single file per night)
- compression of the resulting parquet files


## Installation

### Requirements
The package was tested with the pinned versions in the `requirements.txt` file.
If something does not work, try to install this exact versions. I would particularly advise 
to use matching or more recent versions of PyArrow and Pandas (version 2.0 is important
as its using underlying Arrow datastructures itself, thus it will break with anything
below 2.0, as far as I'm aware).

```bash
pip install git+https://github.com/NarayanSchuetz/edf2parquet.git
```

## Usage
### Converting: 
Convert an EDF file into Apache parquet format using the EdfToParquetConverter class directly
```python
import pytz

from edf2parquet.converters import EdfToParquetConverter, AdvancedEdfToParquetConverter

my_edf_file_path = "path_to_my_edfile.edf"
my_parquet_output_dir = "path_to_my_parquet_output_dir"

converter = EdfToParquetConverter(edf_file_path=my_edf_file_path,
                                  datetime_index=True,
                                  local_timezone=(pytz.timezone("Europe/Zurich"), pytz.timezone("Europe/Zurich")),
                                  parquet_output_dir=my_parquet_output_dir,
                                  compression_codec="GZIP")

converter.convert()

# or alternatively using the advanced converter
converter = AdvancedEdfToParquetConverter(edf_file_path=my_edf_file_path,  # path to the EDF file
                                          exclude_signals=["Audio"],  # list of signals to exclude from the conversion
                                          parquet_output_dir=my_parquet_output_dir,  # path to the output directory (will be created if not exists)
                                          group_by_sampling_freq=True,  # whether to group signals with same sampling frequency into single parquet files
                                          datetime_index=True,  # whether to automatically add a pd.DatetimeIndex to the resulting parquet files
                                          local_timezone=(pytz.timezone("Europe/Zurich"), pytz.timezone("Europe/Zurich")),  # specifies the timezone of the EDF file and the timezone of the start_date in the EDF file (should be the same for most cases)
                                          compression_codec="GZIP", # compression codec to use for the resulting parquet files
                                          split_non_use_by_col="MY_COLUMN")  # only specify this if you want to split the EDF file by non-use periods (e.g. if a file consists of continuous multiple nights and you want to split it into a single file per night) -> read the docstring of the AdvancedEdfToParquetConverter class for more information. The column specifies the column to use for splitting the file.

converter.convert()
```
### Reading:
#### Using standard libraries (e.g. Pandas)
A converted parquet file can be read with any Apache Parquet compatible library, e.g. pandas (using pyarrow engine).
```python
import pandas as pd
my_parquet_file_path = "edf2parquet/tests/test_resources/EEG T5-LE.parquet"

df = pd.read_parquet("edf2parquet/tests/test_resources/EEG T5-LE.parquet")
```
#### Using the provided ParquetEdfReader directly
Using the provided ParquetEdfReader one may read back the EDF associated metadata and directly apply timezone 
conversions given the information was provided during conversion.

```python
from edf2parquet.readers import ParquetReader

my_parquet_file_path = "edf2parquet/tests/test_resources/EEG T5-LE.parquet"
reader = ParquetReader(parquet_file_path=my_parquet_file_path)

# the same as using pandas but with automatic timezone conversion from UTC to local timezone.
df = reader.get_pandas_dataframe(
    set_timezone=True)  # (Note that here we set a timezone which is different to when we used plain pandas)

# however, we cal also get EDF file header information.
reader.get_file_header()

# ... as well as the signal headers
reader.get_signal_headers()
```

Check the `examples.ipynb` notebook for detailed outputs.

## Todo
- [x] Allow to bundle signals with the same sampling rate into a single parquet file.
- [ ] Provide a high level user API
- [ ] Enable (possibly distributed) parallel processing to efficiently convert a whole directory of EDF files.

