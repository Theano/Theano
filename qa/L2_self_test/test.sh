#!/bin/bash

cd /opt/theano

THEANO_FLAGS="floatX=float32" exec theano-test --verbose
