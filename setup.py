
from setuptools import setup, find_packages

MAJOR = "0"
MINOR = "0"
PATCH = "1"

_VERSION_TAG = "{MAJOR}.{MINOR}.{PATCH}".format(MAJOR=MAJOR, MINOR=MINOR, PATCH=PATCH)

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_version():
    import subprocess
    commit_hash = str(subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout)[2:-3]
    return '{VERSION_TAG}.{GIT_SHA}'.format(VERSION_TAG=_VERSION_TAG, GIT_SHA=commit_hash)


setup(
    name="edf2parquet",
    version=get_version(),
    author="Narayan Schütz",
    author_email="narayan.schuetz@gmail.com",
    description="A Python based utility package to convert EDF/EDF+ files to Apache Parquet file format.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NarayanSchuetz/edf2parquet",
    packages=find_packages(),
    install_requires=[
        'pytest',
        'numpy',
        'pandas',
        'pyarrow',
        'pyedflib==0.1.19',  # newer versions
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest-runner',
        'pytest',
        'pytest-mock',
        'requests'
    ],
    zip_safe=False,
    python_requires='>=3.8'
)