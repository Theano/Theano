#!/usr/bin/env python
#
#  TODO:
#   * Figure out how to compile and install documentation automatically
#   * Add download_url

from __future__ import absolute_import, print_function, division
import os
import subprocess
import codecs
from fnmatch import fnmatchcase
from distutils.util import convert_path
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Topic :: Software Development :: Code Generators
Topic :: Software Development :: Compilers
Topic :: Scientific/Engineering :: Mathematics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
"""
NAME                = 'Theano'
MAINTAINER          = "LISA laboratory, University of Montreal"
MAINTAINER_EMAIL    = "theano-dev@googlegroups.com"
DESCRIPTION         = ('Optimizing compiler for evaluating mathematical ' +
                       'expressions on CPUs and GPUs.')
LONG_DESCRIPTION    = (codecs.open("DESCRIPTION.txt", encoding='utf-8').read() +
                       "\n\n" + codecs.open("NEWS.txt", encoding='utf-8').read())
URL                 = "http://deeplearning.net/software/theano/"
DOWNLOAD_URL        = ""
LICENSE             = 'BSD'
CLASSIFIERS         = [_f for _f in CLASSIFIERS.split('\n') if _f]
AUTHOR              = "LISA laboratory, University of Montreal"
AUTHOR_EMAIL        = "theano-dev@googlegroups.com"
PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"]
MAJOR               = 0
MINOR               = 9
MICRO               = 0
SUFFIX              = "rc1"  # Should be blank except for rc's, betas, etc.
ISRELEASED          = False

VERSION             = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)


def find_packages(where='.', exclude=()):
    out = []
    stack = [(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out


def git_version():
    """
    Return the sha1 of local git HEAD as a string.
    """
    # josharian: I doubt that the minimal environment stuff here is
    # still needed; it is inherited. This was originally
    # an hg_version function borrowed from NumPy's setup.py.
    # I'm leaving it in for now because I don't have enough other
    # environments to test in to be confident that it is safe to remove.
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'PYTHONPATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            env=env
        ).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "unknown-git"
    return git_revision


def write_text(filename, text):
    try:
        with open(filename, 'w') as a:
            a.write(text)
    except Exception as e:
        print(e)


def write_version_py(filename=os.path.join('theano', 'generated_version.py')):
    cnt = """
# THIS FILE IS GENERATED FROM THEANO SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
git_revision = '%(git_revision)s'
full_version = '%(version)s.dev-%%(git_revision)s' %% {
    'git_revision': git_revision}
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULL_VERSION = VERSION
    if os.path.isdir('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        GIT_REVISION = "RELEASE"
    else:
        GIT_REVISION = "unknown-git"

    FULL_VERSION += '.dev-' + GIT_REVISION
    text = cnt % {'version': VERSION,
                  'full_version': FULL_VERSION,
                  'git_revision': GIT_REVISION,
                  'isrelease': str(ISRELEASED)}
    write_text(filename, text)


def do_setup():
    write_version_py()
    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          url=URL,
          license=LICENSE,
          platforms=PLATFORMS,
          packages=find_packages(),
          install_requires=['numpy>=1.9.1', 'scipy>=0.14', 'six>=1.9.0'],
          # pygments is a dependency for Sphinx code highlight
          extras_require={
              'test': ['nose>=1.3.0', 'nose-parameterized>=0.5.0', 'flake8<3'],
              'doc': ['Sphinx>=0.5.1', 'pygments']
          },
          package_data={
              '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.c', '*.sh', '*.pkl',
                   '*.h', '*.cpp', 'ChangeLog'],
              'theano.misc': ['*.sh'],
              'theano.d3viz' : ['html/*','css/*','js/*']
          },
          entry_points={
              'console_scripts': ['theano-cache = bin.theano_cache:main',
                                  'theano-nose = bin.theano_nose:main',
                                  'theano-test = bin.theano_test:main']
          },
          keywords=' '.join([
              'theano', 'math', 'numerical', 'symbolic', 'blas',
              'numpy', 'gpu', 'autodiff', 'differentiation'
          ]),
    )
if __name__ == "__main__":
    do_setup()
