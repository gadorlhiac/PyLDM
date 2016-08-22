#!/usr/bin/python

from setuptools import setup

setup(name='PyLDA',
      version='0.1',
      description='Python Based Lifetime Density Analysis',
      author='Gabriel Dorlhiac',
      author_email='gadorlhiac@gmail.com',
      packages=['','.Fit'],
      requires=['matplotlib','numpy','scipy'],
      package_data={'PyLDA':['data/*.dat']},
      package_dir={'':'src/'},
     )

