import setuptools
from distutils.command.build_clib import build_clib
from distutils.core import Extension, setup

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# os.environ["CXX"] = "g++-7"

__version__ = "0.1.1"

requirements = [
    'numpy'
]

setup_requires = [
    'cython'
]

libcnpy = ('cnpy', {'sources': ['cnpy/cnpy.cpp']})

ext_modules = cythonize([
    Extension(
        "xseis2.xspy",  # the extension name
        sources=["xseis2/xspy.pyx"],  # the Cython, cpp source
        language="c++",
        extra_compile_args=[
            "-std=c++17", "-O3", "-Wall", "-fno-wrapv", "-fno-strict-aliasing",
            "-lfftw3f", "-lm", "-pthread", "-march=native", "-ffast-math",
            "-lm", "-fopenmp", "-lz", "-lzip"
        ],
        extra_link_args=["-fopenmp"],
        libraries=[
            'fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads', 'fftw3f_threads',
            'fftw3l_threads', 'z'
        ],
        include_dirs=[np.get_include(), 'xseis2/include', 'cnpy'],
    )
])


def main():
    setup(
        name='xseis2',
        install_requires=requirements,
        libraries=[libcnpy],
        version=__version__,
        cmdclass={
            'build_clib': build_clib,
            'build_ext': build_ext
        },
        packages=setuptools.find_packages(exclude=['tests']),
        ext_modules=ext_modules)


if __name__ == '__main__':
    main()
