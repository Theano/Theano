#!/usr/bin/env python
#
#  TODO:
#   * Figure out how to compile and install documentation automatically
#   * Add back in installation requirements
#   * Add download_url

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

setup(name="Theano",
      version="0.1",
      description="Optimizing compiler for mathematical expressions",
      long_description="""Theano is a Python library that allows you to define, optimize, and efficiently evaluate mathematical expressions involving multi-dimensional arrays. Using Theano, it is not uncommon to see speed improvements of ten-fold over using pure NumPy.""",
      author="LISA laboratory, University of Montreal",
      author_email="theano-dev@googlegroups.com",
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      url="http://pylearn.org/theano",
      keywords="machine learning ai gradient compiler math science",

      test_suite = "nose.collector",

#      install_requires = ["numpy>=1.2", "scipy"],

      extras_require = {
        'doc':      ["sphinx>=0.5.1", "pygments"],
        'test':     ["nose"],
#        'sparse':   ["scipy>=0.7"]
      },
      license = "BSD",
)

