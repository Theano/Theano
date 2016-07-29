#!/bin/bash

# Script for Jenkins continuous integration pre-testing

source ~/.bashrc

# Test flake8
echo "\n===== Testing flake8"
bin/theano-nose theano/tests/test_flake8.py || exit 1

# Test documentation
echo "\n===== Testing documentation build"
python doc/scripts/docgen.py --nopdf --check || exit 1
echo "\n===== Testing documentation code snippets"
python doc/scripts/docgen.py --test --check || exit 1
