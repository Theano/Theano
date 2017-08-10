#!/bin/bash

# curl -fsSL -O http://deeplearning.net/tutorial/code/imdb.py
# curl -fsSL -O http://deeplearning.net/tutorial/code/lstm.py
export THEANO_FLAGS="floatX=float32,mode=FAST_RUN"

exec python lstm.py
