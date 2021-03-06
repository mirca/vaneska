#!/usr/bin/env python
import sys
from setuptools import setup
# To use a consistent encoding
from codecs import open
import os
from os import path

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/vaneska*")
    sys.exit()

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vaneska',
    packages=['vaneska'],
    version='0.0dev',
    description='Sweet PSF photometry tool for TESS',
    long_description=long_description,
    url='https://github.com/mirca/vaneska',
    author='mirca',
    author_email='jvmirca@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    tests_require=['pytest', 'pytest-cov'],
    keywords='statistics probability',
    install_requires=['numpy', ['tensorflow']]
)
