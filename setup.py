#!/bin/env python

from setuptools import setup, find_packages

setup(name="Theano",
      version="0.1",
      description="Machine learning toolkit",
      author="LISA laboratory, Université de Montréal",
      author_email="theano-dev@googlegroups.com",
      packages=find_packages(exclude='tests'),
      url="http://lgcm.iro.umontreal.ca/theano",
      license = "BSD",
)

