# Conversion from Markdown to pypi's restructured text: https://coderwall.com/p/qawuyq -- Thanks James.

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = ''
    print 'warning: pandoc or pypandoc does not seem to be installed; using empty long_description'

import importlib
from pip.req import parse_requirements
from setuptools import setup

install_requires = parse_requirements('requirements.txt', session=False)
install_requires = [str(ir.req) for ir in install_requires]

setup(
    name='example',
    version=importlib.import_module('example').__version__,
    author='Body Labs',
    author_email='___@bodylabs.com',
    description='___',
    long_description=long_description,
    url='https://github.com/bodylabs/___',
    license='MIT',
    packages=[
        'example',
        'example/util',
    ],
    scripts=[
        'bin/hello',
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
