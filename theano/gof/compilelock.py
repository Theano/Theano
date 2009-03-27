# Locking mechanism.

import compiledir
import os
from plearn.utilities.write_results import lockFile, unlockFile

def get_lock():
    """
    Obtain lock on compilation directory.
    """
    if not hasattr(get_lock, 'n_lock'):
        # Initialization.
        get_lock.n_lock = 0
        get_lock.lock_file = os.path.join(compiledir.get_compiledir(), 'lock')
        if not hasattr(get_lock, 'lock_is_enabled'):
            get_lock.lock_is_enabled = True
    # Only really try to acquire the lock if we do not have it already.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        lockFile(get_lock.lock_file, timeout = 60, verbosity = 10)
    get_lock.n_lock += 1

def release_lock():
    """
    Release lock on compilation directory.
    """
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        unlockFile(get_lock.lock_file)

def set_lock_status(use_lock):
    """
    Enable or disable the lock on the compilation directory (which is enabled
    by default). Disabling may make compilation slightly faster (but is not
    recommended for parallel execution).

    @param use_lock: whether to use the compilation lock or not
    @type  use_lock: bool
    """
    get_lock.lock_is_enabled = use_lock

