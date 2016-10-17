#!/usr/bin/env python

from distutils.core import setup

setup(name='PyLDA',
      version='0.1',
      description='Python Based Lifetime Density Analysis of Time-Resolved Ultrafast Data',
      author='Gabriel Dorlhiac, Clyde Fare',
      url='https://www.github.com/gadorlhiac/pylda',
      packages=['pylda', 'pylda.Fit'],

      install_requires=[
          "matplotlib",
          "numpy",
          "scipy",
      ],     
)
