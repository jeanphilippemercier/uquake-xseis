from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
os.environ["CXX"] = "g++-7"


setup(ext_modules=cythonize(Extension(
           "xspy",                                # the extension name
           sources=["xspy.pyx"],  # the Cython, cpp source
           language="c++",
           extra_compile_args=["-std=c++17", "-O3", "-Wall", "-fno-wrapv", "-fno-strict-aliasing", "-lfftw3f", "-lm", "-pthread", "-march=native", "-ffast-math", "-lm", "-fopenmp", "-lcnpy", "-lz"],
           extra_link_args=["-fopenmp"],
           libraries=['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads', 'fftw3f_threads', 'fftw3l_threads', 'cnpy'],
           include_dirs=[np.get_include(), 'include'],
      )))

# time g++-7 $fullfile -std=c++17 -Wall -o temp -O3 -march=native -ffast-math -lhdf5_serial -lhdf5_cpp -pthread -lfftw3f -lm -fopenmp -lstdc++fs -lcnpy -lz && time ./temp
