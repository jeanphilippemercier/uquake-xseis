#!/bin/bash
rm *.so
python3 setup.py build_ext --inplace
cp *.so ../pyinclude/xseis2/
