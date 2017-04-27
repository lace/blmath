# Conversion from Markdown to pypi's restructured text: https://coderwall.com/p/qawuyq -- Thanks James.

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''
    print 'warning: pandoc or pypandoc does not seem to be installed; using empty long_description'

import importlib
import platform
from pip.req import parse_requirements
from setuptools import setup, Extension
import numpy as np


install_requires = parse_requirements('requirements.txt', session=False)
install_requires = [str(ir.req) for ir in install_requires]

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
    name='blmath',
    version=importlib.import_module('blmath').__version__,
    author='Body Labs',
    author_email='alex@bodylabs.com',
    description='A collection of math related utilities used by many bits of BodyLabs code',
    long_description=long_description,
    url='https://github.com/bodylabs/blmath',
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
        'Programming Language :: Python',
    ]
)
