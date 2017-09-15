#!/bin/bash

# Script for Jenkins continuous integration testing on macOS

# Print commands as they are executed
set -x

# Copy cache from master
BASECOMPILEDIR=$HOME/.theano/pr_theano_mac
# rsync -a --delete $HOME/.theano/buildbot_theano_mac/ $BASECOMPILEDIR

# Set path for conda and cmake
export PATH="/Users/jenkins/miniconda2/bin:/usr/local/bin:$PATH"

# Testing theano
THEANO_PARAM="theano -e gpuarray --with-timer --timer-top-n 10 --with-xunit --xunit-file=theano_mac_pr_tests.xml"
FLAGS=mode=FAST_RUN,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800,base_compiledir=$BASECOMPILEDIR
THEANO_FLAGS=${FLAGS} python bin/theano-nose ${THEANO_PARAM}
