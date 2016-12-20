#!/usr/bin/env python
import setuptools
from distutils.core import setup

setup(name='PyLDA',
      version='0.1',
      description='Python Based Lifetime Density Analysis of Time-Resolved Ultrafast Data',
      author='Gabriel Dorlhiac, Clyde Fare',
      url='https://www.github.com/gadorlhiac/pylda',
      package_dir={'':'pylda'},
      packages=['pylda', 'pylda.fit', 'pylda.test'],
      package_data={'pylda' : ['data/*.csv']},

      install_requires=[
          "matplotlib",
          "numpy",
          "scipy",
      ],     
)
