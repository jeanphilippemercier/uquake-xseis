#!/bin/bash
rm *.so
# python setup.py build_ext --inplace
python3 setup.py build_ext --inplace
cp *.so ../pyinclude/xseis2/.
