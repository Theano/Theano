#!/usr/bin/env python
#
#  TODO:
#   * Figure out how to compile and install documentation automatically
#   * Add back in installation requirements
#   * Add download_url


from distutils.core import setup

setup(name='Theano',
            version='hg',
            description='Optimizing compiler for mathematical expressions',
            author='LISA laboratory, University of Montreal',
            author_email='theano-dev@googlegroups.com',
            url='http://www.deeplearning.net/software/theano',
            packages=['theano', 'theano.tensor', 'theano.gof',
                      'theano.compile', 'theano.misc', 'theano.scalar',
                      'theano.sparse', 
                      'theano.tensor.nnet', 'theano.tensor.signal'],
           )
