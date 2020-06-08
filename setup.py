#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='src',
      version='0.0.1',
      description='Non-Autoregressive Task-Oriented System',
      author='Yen-Ting (Adam), Lin',
      author_email='r08944064@csie.ntu.edu.tw',
      url='https://github.com/adamlin120/NATODS',
      install_requires=[
            'torch'
            'pytorch-lightning'
      ],
      packages=find_packages()
      )
