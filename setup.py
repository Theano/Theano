#!/usr/bin/env python
#
#  TODO:
#   * Figure out how to compile and install documentation automatically
#   * Add download_url

# Detect whether or not the user has setuptools and use the bundled
# distribute_setup.py bootstrap module if they don't.
try:
    from setuptools import setup, find_packages
except ImportError:
    import distribute_setup
    distribute_setup.use_setuptools()
    from setuptools import setup, find_packages

import os
import subprocess

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
MINOR               = 4
MICRO               = 0
SUFFIX              = "rc3"  # Should be blank except for rc's, betas, etc.
ISRELEASED          = False

VERSION             = '%d.%d.%d%s' % (MAJOR, MINOR, MICRO, SUFFIX)

# Return the hg revision as a string -- borrowed from hg_version in NumPy's
# setup.py file
def hg_version():
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
            stdout = subprocess.PIPE,
            env=env
        ).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['hg', '-q', 'id'])
        HG_REVISION = out.strip().decode('ascii')
    except OSError:
        HG_REVISION = "unknown-hg"
    return HG_REVISION

def write_version_py(filename='theano/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM THEANO SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
hg_revision = '%(hg_revision)s'
full_version = '%(version)s.dev-%%(hg_revision)s' %% {'hg_revision': hg_revision}
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULL_VERSION = VERSION
    if os.path.exists('.hg'):
        HG_REVISION = hg_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        HG_REVISION = "RELEASE"
    else:
        HG_REVISION = "unknown-hg"

    FULL_VERSION += '.dev-' + HG_REVISION

    a = open(filename, 'w')
    try:
    	try:
        	a.write(cnt % {'version': VERSION,
                       	'full_version' : FULL_VERSION,
                       	'hg_revision' : HG_REVISION,
                       	'isrelease': str(ISRELEASED)})
    	except Exception, e:
        	print e
    finally:
        a.close()


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
          install_requires=['numpy>=1.3.0', 'scipy>=0.7.0'],
          package_data={
              '': ['*.txt', '*.rst', '*.cu', '*.cuh', '*.sh', 'ChangeLog'],
              'theano.misc': ['*.sh']
          },
          keywords=' '.join([
            'theano', 'math', 'numerical', 'symbolic', 'blas',
            'numpy', 'gpu', 'autodiff', 'differentiation'
          ])
    )
if __name__ == "__main__":
    do_setup()
