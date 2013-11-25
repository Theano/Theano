#!/usr/bin/env python
#
#  TODO:
#   * Figure out how to compile and install documentation automatically
#   * Add download_url

import os
import sys
import subprocess
from fnmatch import fnmatchcase
from distutils.util import convert_path
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py
    from distutils.command.build_scripts import build_scripts
else:
    exclude_fixers = ['fix_next', 'fix_filter']
    from distutils.util import Mixin2to3
    from lib2to3.refactor import get_fixers_from_package
    Mixin2to3.fixer_names = [f for f in get_fixers_from_package('lib2to3.fixes')
                             if f.rsplit('.', 1)[-1] not in exclude_fixers]
    from distutils.command.build_scripts import build_scripts_2to3 as build_scripts


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
Programming Language :: Python :: 2.4
Programming Language :: Python :: 2.5
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
"""
NAME                = 'Theano'
MAINTAINER          = "LISA laboratory, University of Montreal"
MAINTAINER_EMAIL    = "theano-dev@googlegroups.com"
DESCRIPTION         = ('Optimizing compiler for evaluating mathematical ' +
                       'expressions on CPUs and GPUs.')
LONG_DESCRIPTION    = (open("DESCRIPTION.txt").read() + "\n\n" +
                       open("NEWS.txt").read())
URL                 = "http://deeplearning.net/software/theano/"
DOWNLOAD_URL        = ""
LICENSE             = 'BSD'
CLASSIFIERS         = filter(None, CLASSIFIERS.split('\n'))
AUTHOR              = "LISA laboratory, University of Montreal"
AUTHOR_EMAIL        = "theano-dev@googlegroups.com"
PLATFORMS           = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"]
MAJOR               = 0
MINOR               = 6
MICRO               = 0
SUFFIX              = "rc5"  # Should be blank except for rc's, betas, etc.
ISRELEASED          = False

VERSION             = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)


def find_packages(where='.', exclude=()):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where,name)
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

# Python 2.4 compatibility: Python versions 2.6 and later support new
# exception syntax, but for now we have to resort to exec. 
if sys.hexversion >= 0x2070000:
    exec("""\
def write_text(filename, text):
    with open(filename, 'w') as a:
        try:
            a.write(text)
        except Exception as e:
            print(e)
""")
else:
    exec("""\
def write_text(filename, text):
    a = open(filename, 'w')
    try:
        try:
            a.write(text)
        except Exception, e:
            print e
    finally:
        a.close()
""")

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
          install_requires=['numpy>=1.5.0', 'scipy>=0.7.2'],
          package_data={
              '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.c', '*.sh', '*.pkl',
                   'ChangeLog'],
              'theano.misc': ['*.sh']
          },
          scripts=['bin/theano-cache', 'bin/theano-nose', 'bin/theano-test'],
          keywords=' '.join([
            'theano', 'math', 'numerical', 'symbolic', 'blas',
            'numpy', 'gpu', 'autodiff', 'differentiation'
          ]),
          cmdclass = {'build_py': build_py,
                      'build_scripts': build_scripts}
    )
if __name__ == "__main__":
    do_setup()
