#!/usr/bin/env python
import setuptools
from distutils.core import setup

setup(name='PyLDM',
      version='0.1',
      description='Python Based Lifetime Density Analysis of Time-Resolved Ultrafast Data',
      author='Gabriel Dorlhiac, Clyde Fare',
      author_email='gadorlhiac@gmail.com',
      url='https://www.github.com/gadorlhiac/pyldm',
      packages=['pyldm'],
      package_data={'pyldm' : ['data/*.csv']},
)
