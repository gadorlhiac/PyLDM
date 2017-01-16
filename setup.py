#!/usr/bin/env python
import setuptools
from distutils.core import setup

setup(name='PyLDM',
      version='0.1',
      description='Python Based Lifetime Density Analysis of Time-Resolved Ultrafast Data',
      author='Gabriel Dorlhiac, Clyde Fare',
      url='https://www.github.com/gadorlhiac/pylda',
      packages=['pylda'],
      package_data={'pylda' : ['data/*.csv']},
)
