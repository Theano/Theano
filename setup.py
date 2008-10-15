#!/bin/env python

from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages, Extension, Library
setup(name="Theano",
      version="0.1",
      description="Theano",
      long_description="""Machine learning toolkit""",
      author="LISA",
      author_email="theano-dev@googlegroups.com",
      packages=find_packages(exclude='tests'),
)

