#!/bin/sh
# Script to update version.py in response to Mercurial hooks. This should
# not appear in a release tarball.

echo "Updating version.py..."
sed -e "s/^hg_revision.*/hg_revision = '`expr substr $HG_NODE 1 12`'/" theano/version.py >theano/version.py.out && mv theano/version.py.out theano/version.py

