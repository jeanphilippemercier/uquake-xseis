FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
ENV CONAN_REVISIONS_ENABLED=1
ENV LD_LIBRARY_PATH=/src/xseis/cnpy/
RUN apt-get -y update; apt-get install python3 -y; apt-get install python3-pip -y;  \
    pip install cython;  \
    apt install libhdf5-dev -y; apt-get install libfftw3-dev libfftw3-doc -y;  \
    apt-get install software-properties-common -y;\
    add-apt-repository ppa:ubuntu-toolchain-r/tests -y; apt-get update; apt install git -y; \
    pip install ipython; pip install python3-tk; pip install useis

RUN apt install apt-utils -y; apt install cmake -y

ADD . /src/xseis
WORKDIR /src/xseis

RUN cd /src/xseis; git submodule update --init; cd cnpy; cmake .; make; make install

RUN CC=g++ CXX=g++ CFLAGS="-I/usr/include/hdf5/serial" pip install -e .


