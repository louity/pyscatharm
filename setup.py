#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = """
Fast CPU/CUDA Solid Harmonic 3D Scattering implementation

Numpy + PyTorch + FFTW / cuFFT implementation
"""

setup_info = dict(
    # Metadata
    name='scatharm',
    version=VERSION,
    author='Louis Thiry',
    author_email='louis(dot)thiry<at>outlook(dot)fr',
    url='https://github.com/louity/pyscatharm',
    description='Fast CPU/CUDA Solid Harmonic 3D Scattering implementation',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,

    install_requires=[
        'torch',
        'six'
    ]
)

setup(**setup_info)
