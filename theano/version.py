from __future__ import absolute_import, print_function, division

from theano._version import get_versions

FALLBACK_VERSION = "1.0.1+unknown"

info = get_versions()
if info['error'] is not None:
    info['version'] = FALLBACK_VERSION

full_version = info['version']
git_revision = info['full-revisionid']
del get_versions

short_version = full_version.split('+')[0]


# This tries to catch a tag like beta2, rc1, ...
try:
    int(short_version.split('.')[2])
    release = True
except ValueError:
    release = False

if release and info['error'] is None:
    version = short_version
else:
    version = full_version
del info
