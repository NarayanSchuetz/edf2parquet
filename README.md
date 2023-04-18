# edf2parquet
Simple utility package to convert EDF/EDF+ files into Apache Parquet format 
while preserving the EDF file header information and signal headers metadata information.

## Installation

```bash
pip install git+https://github.com/NarayanSchuetz/edf2parquet.git
```

## Usage
### Converting: 
Convert an EDF file into Apache parquet format using the EdfToParquetConverter class directly
```python
import pytz

from edf2parquet.converters import EdfToParquetConverter

my_edf_file_path = "path_to_my_edfile.edf"
my_parquet_output_dir = "path_to_my_parquet_output_dir"

converter = EdfToParquetConverter(edf_file_path=my_edf_file_path,
                                  datetime_index=True,
                                  local_timezone=(pytz.timezone("Europe/Zurich"), pytz.timezone("Europe/Zurich")),
                                  parquet_output_dir=my_parquet_output_dir,
                                  compression_codec="GZIP")

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
from edf2parquet.readers import ParquetEdfReader
my_parquet_file_path = "edf2parquet/tests/test_resources/EEG T5-LE.parquet"
reader = ParquetEdfReader(parquet_file_path=my_parquet_file_path)

# the same as using pandas but with automatic timezone conversion from UTC to local timezone.
df = reader.get_pandas_dataframe(set_timezone=True) # (Note that here we set a timezone which is different to when we used plain pandas)

# however, we cal also get EDF file header information.
reader.get_file_header()

# ... as well as the signal headers
reader.get_signal_headers()
```

Check the `examples.ipynb` notebook for detailed outputs.

## Todo
- [ ] Allow to bundle signals with the same sampling rate into a single parquet file.
- [ ] Provide a high level user API
- [ ] Enable (possibly distributed) parallel processing to efficiently convert a whole directory of EDF files.
- [ ] Provide a high level API to convert EDF files with the same sampling frequency (fs) into a single parquet file with a single row per signal.

