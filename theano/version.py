from __future__ import absolute_import, print_function, division

from theano._version import get_versions

info = get_versions()
full_version = info['version']
git_revision = info['full-revisionid']
del info, get_versions

short_version = full_version.split('+')[0]

# This tries to catch a tag like beta2, rc1, ...
try:
    int(short_version.split('.')[2])
    release = True
except (ValueError, IndexError):
    # IndexError can happen with the version:
    # 0+untagged.2896.g0560a7f for example
    release = False

if release:
    version = short_version
else:
    version = full_version
