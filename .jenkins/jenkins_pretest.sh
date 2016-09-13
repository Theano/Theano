#!/bin/bash

# Script for Jenkins continuous integration pre-testing

# Print commands as they are executed
set -x

# Anaconda python
export PATH=/usr/local/miniconda2/bin:$PATH

# Test flake8
echo "===== Testing flake8"
bin/theano-nose theano/tests/test_flake8.py --with-xunit --xunit-file=theano_pre_tests.xml || exit 1

# Test documentation
echo "===== Testing documentation build"
python doc/scripts/docgen.py --nopdf --check || exit 1
echo "===== Testing documentation code snippets"
python doc/scripts/docgen.py --test --check || exit 1
