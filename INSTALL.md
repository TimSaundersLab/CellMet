## System requirements
Linux, Windows and Mac OS are supported for running the code. At least 8 GB of RAM is required to run the software. 32GB may be required for larger images. The software has been heavily tested on Ubuntu 22.04 and less-tested on Mac OS and Windows 10. Please open an issue if you have any problems with installation. 


## Installing CellSeg with conda

If you have a conda environment ready:
```
conda install -c conda-forge CellSeg
```

This will install CellSeg and all its dependencies, with the pre-compiled binary parts.

## Install CellSeg using pip

This install a cross-platform, pure python version of CellSeg.
Some advanced features are not available, namely:

```sh
python -m pip install --user --upgrade CellSeg
```

## Installing from source

Those are the instructions to install the package from source on a
debian-like linux distribution. If you already have a basic
scientific python stack, use it, don't install anaconda.


### Download and install `CellSeg` from source

If you want to do that, I assume you already know how to manage
dependencies on your platform. The simplest way to manage dependencies is to use [`conda`](https://docs.conda.io/en/latest/miniconda.html) to manage the dependencies (you can use [`mamba`](https://github.com/mamba-org/mamba) as a faster alternative to conda).


```bash
git clone --recursive https://github.com/sophietheis/CellSeg.git
cd CellSeg
```

Then create a virtual environement:

```bash
conda env create -f environment.yml
```

Then install python:
```
python setup.py install
```

Or

```
pip install .
```



If all went well, you have successfully installed CellSeg.

