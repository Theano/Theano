#!/usr/bin/env bash
# In Python 3.4, we test the min version of NumPy and SciPy. In Python 2.7, we test more recent version.
if test -e $HOME/miniconda2/envs/pyenv; then
    echo "pyenv already exists."
else
    echo "Creating pyenv."
    if [[ $TRAVIS_PYTHON_VERSION == '2.7' ]]; then conda create --yes -q -n pyenv python=2.7 ; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.4' ]]; then conda create --yes -q -n pyenv python=3.4 ; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.6' ]]; then conda create --yes -q -n pyenv python=3.6 ; fi
fi

source activate pyenv
if [[ $TRAVIS_PYTHON_VERSION == '2.7' && $NUMPY_VERSION == '1.9.1' ]]; then conda install --yes -q mkl numpy=1.9.1 scipy=0.14.0 nose=1.3.0 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx=1.5.1 mkl-service libgfortran=1 graphviz; fi
if [[ $TRAVIS_PYTHON_VERSION == '2.7' && $NUMPY_VERSION == '1.13.1' ]]; then conda install --yes -q mkl numpy=1.13.1 scipy=0.19.1 nose=1.3.0 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx=1.5.1 mkl-service libgfortran=3 graphviz; fi
if [[ $TRAVIS_PYTHON_VERSION == '3.4' && $NUMPY_VERSION == '1.9.1' ]]; then conda install --yes -q mkl numpy=1.9.1 scipy=0.14.0 nose=1.3.4 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx=1.5.1 mkl-service libgfortran=1 graphviz pygments=2.1.3; fi
if [[ $TRAVIS_PYTHON_VERSION == '3.6' && $NUMPY_VERSION == '1.13.1' ]]; then conda install --yes -q mkl numpy=1.13.1 scipy=0.19.1 nose=1.3.7 pip flake8=3.5 six=1.11.0 pep8=1.7.1 pyflakes=1.6.0 sphinx=1.5.1 mkl-service libgfortran=3 graphviz; fi
source deactivate
