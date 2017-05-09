#!/usr/bin/env bash

## This command should make this script stop at first error occured
## or keyboard interruption received.
set -e

## Get all DNN algorithms supported by Theano.

export SUPPORTED_DNN_CONV_ALGO_FWD=$(python -c "from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME, SUPPORTED_DNN_CONV_ALGO_FWD; print('\n'.join(SUPPORTED_DNN_CONV_ALGO_FWD + SUPPORTED_DNN_CONV_ALGO_RUNTIME))")

export SUPPORTED_DNN_CONV_ALGO_BWD_DATA=$(python -c "from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME, SUPPORTED_DNN_CONV_ALGO_BWD_DATA; print('\n'.join(SUPPORTED_DNN_CONV_ALGO_BWD_DATA + SUPPORTED_DNN_CONV_ALGO_RUNTIME))")

export SUPPORTED_DNN_CONV_ALGO_BWD_FILTER=$(python -c "from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME, SUPPORTED_DNN_CONV_ALGO_BWD_FILTER; print('\n'.join(SUPPORTED_DNN_CONV_ALGO_BWD_FILTER + SUPPORTED_DNN_CONV_ALGO_RUNTIME))")

export SUPPORTED_DNN_CONV_PRECISION=$(python -c "from theano.configdefaults import SUPPORTED_DNN_CONV_PRECISION; print('\n'.join(SUPPORTED_DNN_CONV_PRECISION))")

## List special Theano DNN tests that depend on DNN algorithms.
## Other Theano DNN tests will be run once.
## These special Theano DNN tests will be run 
## for every combination of DNN algorithms.

declare -a special_tests=('theano.gpuarray.tests.test_dnn.test_conv3d_bwd'
                          'theano.gpuarray.tests.test_dnn.test_conv3d_fwd'
                          'theano.gpuarray.tests.test_dnn.test_dnn_conv_alpha_output_merge'
                          'theano.gpuarray.tests.test_dnn.test_dnn_conv_border_mode'
                          'theano.gpuarray.tests.test_dnn.test_dnn_conv_grad'
                          'theano.gpuarray.tests.test_dnn.test_dnn_conv_inplace'
                          'theano.gpuarray.tests.test_dnn.test_dnn_conv_merge'
                          'theano.gpuarray.tests.test_dnn.TestDnnInferShapes')

## Precompute nosetests arguments used later.

# PB: To test quickly this script, you can temporarly 
# add "--collect-only" arg into this variable.
export NOSEARGS="-xvs --collect-only" #export NOSEARGS="-xvs"

# Convert list of special tests to a list of tests to be executed by nosetests.
export SPECIAL_NOSETESTS=$(echo ${special_tests[@]} | sed 's/\.test_dnn\./.test_dnn:/g')

# Convert list of special tests to a list of tests to be excluded by nosetests.
export SPECIAL_NOSETESTS_EXCLUDED=$(for test in "${special_tests[@]}"; do echo "--exclude-test=$test"; done)


## Run all Theano DNN tests without special tests.

echo Running independent DNN tests.

THEANO_FLAGS=warn.dnn_recent=False eval "nosetests $NOSEARGS $(echo $SPECIAL_NOSETESTS_EXCLUDED) theano/gpuarray/tests/test_dnn.py"

## Run all special Theano DNN tests for every combination of DNN algorithms.
# This may take a very (very) long time.

echo Running DNN tests depending on algorithms parameters.

for fwd in $SUPPORTED_DNN_CONV_ALGO_FWD
do
    for bwd_filter in $SUPPORTED_DNN_CONV_ALGO_BWD_FILTER
    do
        for bwd_data in $SUPPORTED_DNN_CONV_ALGO_BWD_DATA
        do
            for precision in $SUPPORTED_DNN_CONV_PRECISION
            do
                # THEANO_FLAGS is set on multiple lines to avoid a declaration with one too long line.
                export THEANO_FLAGS=warn.dnn_recent=False
                export THEANO_FLAGS=$THEANO_FLAGS,dnn.conv.algo_fwd=$fwd
                export THEANO_FLAGS=$THEANO_FLAGS,dnn.conv.algo_bwd_data=$bwd_data
                export THEANO_FLAGS=$THEANO_FLAGS,dnn.conv.algo_bwd_filter=$bwd_filter
                export THEANO_FLAGS=$THEANO_FLAGS,dnn.conv.precision=$precision
                echo "Running with: $THEANO_FLAGS"
                eval "nosetests $NOSEARGS $SPECIAL_NOSETESTS"
            done
        done
    done
done
