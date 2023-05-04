FROM ubuntu:latest
ENV DEBIAN_FRONTEND noninteractive
ENV CONAN_REVISIONS_ENABLED=1
ENV LD_LIBRARY_PATH=/src/xseis/cnpy/
RUN apt-get -y update; apt-get install python3 -y; apt-get install python3-pip -y;  \
    pip install cython;  \
    apt install libhdf5-dev -y; apt-get install libfftw3-dev libfftw3-doc -y;  \
    apt-get install software-properties-common -y;\
    add-apt-repository ppa:ubuntu-toolchain-r/test -y; apt-get update; apt install git -y;

RUN apt install apt-utils -y; apt install cmake -y

# Install Conan
RUN apt-get update && \
    apt-get install -y python3-setuptools && \
    pip3 install conan && \
    mkdir -p /root/.conan/profiles && \
    conan profile detect
#    conan profile show default > /root/.conan2/profiles/default && \
#    sed -i 's/compiler.version=/compiler.version=7/' /root/.conan/profiles/new_profile

RUN apt-get install python3-tk; pip install numpy; pip install obspy;

#RUN pip install conan &&  \
#    conan remote add ecdc https://artifactoryconan.esss.dk/artifactory/api/conan/ecdc &&  \
#    conan remote add bincrafters https://bincrafters.jfrog.io/artifactory/api/conan/public-conan

ADD . /src/xseis
WORKDIR /src/xseis


RUN cd /src/xseis; git submodule update --init; cd cnpy; cmake .; make; make install

RUN CC=g++ CXX=g++ CFLAGS="-I/usr/include/hdf5/serial" pip install -e .


