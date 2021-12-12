FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
ENV CONAN_REVISIONS_ENABLED=1
RUN apt-get -y update; apt-get install python3 -y; apt-get install python3-pip -y; pip install cython;  \
    apt install libhdf5-dev -y; apt-get install libfftw3-dev libfftw3-doc -y; \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y; apt-get update;

#RUN conan remote add ecdc https://artifactoryconan.esss.dk/artifactory/api/conan/ecdc &&  \
#    conan remote add bincrafters https://bincrafters.jfrog.io/artifactory/api/conan/public-conan
# RUN apt-get install cmake -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && apt-get update \
    && apt-get -y install cmake protobuf-compiler && pip install conan && apt-get -y install gcc  \
    && apt-get -y install g++ && apt-get -y install libhdf5-dev
ADD . /src/xseis
WORKDIR /src/xseis

RUN pip install numpy
# RUN CC=g++-7 CXX=g++-7 CFLAGS="-I/usr/include/hdf5/serial" pip install -e .

#RUN pip install obspy; pip install cython; apt install libhdf5-dev -y;  \
#    apt-get install libfftw3-dev libfftw3-doc -y; add-apt-repository ppa:ubuntu-toolchain-r/test -y; \
#    apt-get update;
#
#WORKDIR /src/xseis/h5cpp
#
#RUN mkdir h5cpp-build
#
#WORKDIR /src/xseis/h5cpp/h5cpp-build
#
#RUN conan user cmake -DCMAKE_BUILD_TYPE=Release ../.
#RUN make; make install
