#!/usr/bin/env bash
# In Python 3.3, we test the min version of NumPy and SciPy. In Python 2.7, we test more recent version.
# nose-exclude plugin should allow use to tell nosetests to exclude folder with --exclude-dir=path/to/directory.
if test -e $HOME/miniconda2/envs/pyenv ; then
    echo "pyenv already exists."
    source activate pyenv
    if [[ $TRAVIS_PYTHON_VERSION == '2.7' ]]; then conda install mkl python=2.7 numpy=1.9.1 scipy=0.14.0 nose=1.3.0 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx mkl-service libgfortran=1; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.3' ]]; then conda install mkl python=3.3 numpy=1.9.1 scipy=0.14.0 nose=1.3.4 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx mkl-service; fi
    source deactivate
else
    echo "Creating pyenv."
    if [[ $TRAVIS_PYTHON_VERSION == '2.7' ]]; then conda create --yes -q -n pyenv mkl python=2.7 numpy=1.9.1 scipy=0.14.0 nose=1.3.0 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx mkl-service libgfortran=1; fi
    if [[ $TRAVIS_PYTHON_VERSION == '3.3' ]]; then conda create --yes -q -n pyenv mkl python=3.3 numpy=1.9.1 scipy=0.14.0 nose=1.3.4 pip flake8=2.3 six=1.9.0 pep8=1.6.2 pyflakes=0.8.1 sphinx mkl-service; fi
fi
