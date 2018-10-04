#!/bin/bash
# rm $XSEISPATH/cython/*.so
rm *.so
python setup.py build_ext --inplace
# python3 setup.py build_ext --build-lib $XSEISPATH/cython/
cp *.so ../pyinclude/xseis2/
# cp $XSEISPATH/cython/*.so $XSEISPATH/pyinclude/xseis2/.
