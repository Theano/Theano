#!/usr/bin/env bash
# In Python 3.3, we test the min version of NumPy and SciPy. In Python 2.7, we test more recent version.
if test -e $HOME/miniconda2/envs/pyenv; then
    echo "pyenv already exists."
else
    echo "Creating pyenv."
    if [[ $TRAVIS_PYTHON_VERSION == '2.7' ]]; then conda create --yes -q -n pyenv python=2.7 ; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.3' ]]; then conda create --yes -q -n pyenv python=3.3 ; fi
fi

source activate pyenv
if [[ $TRAVIS_PYTHON_VERSION == '2.7' ]]; then conda install --yes -q mkl numpy=1.9.1 scipy=0.14.0 nose=1.3.0 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx mkl-service libgfortran=1; fi
if [[ $TRAVIS_PYTHON_VERSION == '3.3' ]]; then conda install --yes -q mkl numpy=1.9.1 scipy=0.14.0 nose=1.3.4 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx mkl-service; fi
source deactivate
