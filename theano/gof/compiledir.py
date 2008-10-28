
import os
import platform
import re

def set_compiledir(path=None):
    """Set the directory into which theano will compile code objects

    @param path: an absolute path or relative path. An argument of None will
    trigger one of two default paths: firstly an environment variable called
    'THEANO_COMPILEDIR' will be sought; failing that, an architecture-specific
    directory will be chosen within $HOME/.theano.

    @type path: string or None

    @return: None

    @note:  This function will create the path (recursively) as a folder if it
    is not present, not readable, or not writable.  New folders will be created
    with mode 0700.

    """
    # N.B. The path is stored as an attribute of this function

    if path is None:
        # we need to set the default, which can come from one of two places
        if os.getenv('THEANO_COMPILEDIR'):
            path = os.getenv('THEANO_COMPILEDIR')
        else:
            platform_id = platform.platform() + '-' + platform.processor()
            platform_id = re.sub("[\(\)\s]+", "_", platform_id)
            path = os.path.join(os.getenv('HOME'), '.theano', 'compiledir_'+platform_id)

    if not os.access(path, os.R_OK | os.W_OK):
        os.makedirs(path, 0770) #read-write-execute for this user only

    # PROBLEM: sometimes the first approach based on os.system('touch')
    # returned -1 for an unknown reason; the alternate approach here worked
    # in all cases... it was weird.
    open(os.path.join(path, '__init__.py'), 'w').close()

    set_compiledir.compiledir = path

def get_compiledir():
    """Return the directory where theano code objects should be compiled

    @rtype: string
    """
    if not hasattr(set_compiledir, 'compiledir'):
        set_compiledir()
    return set_compiledir.compiledir
