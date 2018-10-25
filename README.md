# xseis

## Build

```bash
# pull cnpy dependency
git submodule update --init
# pip install --user .

# in ubuntu you might depend on g++7 or higher
# CC=g++-7 CXX=g++-7 CFLAGS="-I/usr/include/hdf5/serial" pip install --user .
CC=g++-8 CXX=g++-8 CFLAGS="-I/usr/include/hdf5/serial" pip install --user .
```

## Python dependencies

```bash
sudo apt install python3-pip
sudo apt-get install python3-tk
pip3 install ipython
pip3 install obspy
pip3 install cython
pip3 install kafka
```

## Compile dependencies

```bash
sudo apt install libhdf5-dev -y
sudo apt-get install libfftw3-dev libfftw3-doc -y

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install gcc-7 g++-7 -y
```
