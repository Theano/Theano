try:
    from theano.generated_version import *
except ImportError:
    short_version = 'unknown'
    version = 'unknown'
    git_revision = 'unknown'
    full_version = 'unknown'
    release = False
