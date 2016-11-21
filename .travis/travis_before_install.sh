#!/usr/bin/env bash
# Install miniconda to avoid compiling scipy
if ! test -e $HOME/miniconda2/bin ; then
    rm -rf $HOME/miniconda2
    wget -c https://repo.continuum.io/miniconda/Miniconda2-4.1.11-Linux-x86_64.sh -O $HOME/download/miniconda.sh
    chmod +x $HOME/download/miniconda.sh
    $HOME/download/miniconda.sh -b
    export PATH=/home/travis/miniconda2/bin:$PATH
else
    echo "miniconda already installed."
fi