#!/bin/bash

python opencv.py $@
python conv2d.py $@
python scipy_conv.py $@

echo "WARNING the mode is valid for theano and scipy, but opencv use the mode same! Can opencv do the mode full?"
