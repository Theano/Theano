#!/bin/sh
# Script to update version.py in response to Mercurial hooks. This should
# not appear in a release tarball.
if [ -z "$HG_NODE" ] ; then
    echo No HG_NODE, skipping update of theano.__version__
    exit 0 # this seems to be normal sometimes
else
    sed -e "s/^hg_revision.*/hg_revision = '`python -c \"print \\"$HG_NODE\\"[0:12]\"`'/" theano/version.py >theano/version.py.out && mv theano/version.py.out theano/version.py
fi

