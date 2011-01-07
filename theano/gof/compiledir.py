
import errno
import os, sys
import platform
import re

from theano.configparser import config, AddConfigVar, StrParam

def default_compiledirname():
    platform_id = '-'.join([
        platform.platform(),
        platform.processor(),
        platform.python_version()])
    platform_id = re.sub("[\(\)\s]+", "_", platform_id)
    return 'compiledir_' + platform_id

def is_valid_compiledir(path):
    if not os.access(path, os.R_OK | os.W_OK):
        try:
            os.makedirs(path, 0770) #read-write-execute for this user only
        except OSError, e:
            # Maybe another parallel execution of theano was trying to create
            # the same directory at the same time.
            if e.errno != errno.EEXIST:
                return False

    try:
        # PROBLEM: sometimes the first approach based on os.system('touch')
        # returned -1 for an unknown reason; the alternate approach here worked
        # in all cases... it was weird.
        open(os.path.join(path, '__init__.py'), 'w').close()
        if path not in sys.path:
            sys.path.append(path)
    except:
        return False
    
    return True

AddConfigVar('base_compiledir',
        "arch-independent cache directory for compiled modules",
        StrParam(os.path.join(config.home, '.theano'), allow_override=False))

AddConfigVar('compiledir',
        "arch-dependent cache directory for compiled modules",
        StrParam(
            os.path.join(
                os.path.expanduser(config.base_compiledir),
                default_compiledirname()),
            is_valid=is_valid_compiledir,
            allow_override=False))
