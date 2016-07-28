#!/bin/bash

# Script for Jenkins continuous integration pre-testing

source ~/.bashrc

# Test flake8
echo "===== Testing flake8"
bin/theano-nose theano/tests/test_flake8.py
# Test documentation
echo "===== Testing documentation"
python doc/scripts/docgen.py --nopdf --check
python doc/scripts/docgen.py --test --check
