# Conversion from Markdown to pypi's restructured text: https://coderwall.com/p/qawuyq -- Thanks James.

import os

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''
    print('warning: pandoc or pypandoc does not seem to be installed; using empty long_description')

import importlib
import platform

from setuptools import setup, Extension
import numpy as np

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    install_requires = f.readlines()

install_requires = [
    x.strip() for x in install_requires
    if x.strip() and not x.startswith('#')
]

if platform.system() == 'Windows':
    libs = ['suitesparseconfig', 'liblapack', 'libcolamd', 'libamd', 'libcholmod', 'libblas']
    include_dirs = [np.get_include()]
    extra_compile_args = []
elif platform.system() == 'Darwin':
    libs = ['suitesparseconfig', 'lapack', 'colamd', 'amd', 'cholmod']
    include_dirs = [np.get_include()]
    extra_compile_args = ['-O0', '-fno-inline']
else:
    libs = ['lapack', 'colamd', 'amd', 'cholmod']
    extra_compile_args = ['-O0', '-fno-inline']
    include_dirs = [np.get_include(), '/usr/include/suitesparse']

setup(
    name='metablmath',
    version=importlib.import_module('blmath').__version__,
    author='Body Labs',
    author_email='alex@bodylabs.com',
    description='Active fork of blmath, a collection math-related utilities developed at Body Labs',
    long_description=long_description,
    url='https://github.com/metabolize/blmath',
    license='MIT',
    packages=[
        'blmath',
        'blmath/geometry',
        'blmath/geometry/primitives',
        'blmath/geometry/transform',
        'blmath/numerics',
        'blmath/numerics/linalg',
        'blmath/optimization',
        'blmath/optimization/objectives',
        'blmath/util',
    ],
    ext_modules=[
        Extension('blmath.numerics.linalg.cholmod',
            libraries=libs,
            include_dirs=include_dirs,
            sources=['blmath/numerics/linalg/cholmod.c'],
            depends=['blmath/numerics/linalg/cholmod.c'],
            extra_compile_args=extra_compile_args,
        ),
    ],
    install_requires=install_requires,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        # Requires baiji, which does not support python 3.
        # 'Programming Language :: Python :: 3',
    ]
)
