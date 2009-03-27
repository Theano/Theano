# Locking mechanism.

import compiledir
import os, random, time

def get_lock():
    """
    Obtain lock on compilation directory.
    """
    if not hasattr(get_lock, 'n_lock'):
        # Initialization.
        get_lock.n_lock = 0
        get_lock.lock_dir = os.path.join(compiledir.get_compiledir(), 'lock_dir')
        if not hasattr(get_lock, 'lock_is_enabled'):
            get_lock.lock_is_enabled = False # Disabled for now by default.
    # Only really try to acquire the lock if we do not have it already.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        lock(get_lock.lock_dir, timeout = 60, verbosity = 10)
    get_lock.n_lock += 1

def release_lock():
    """
    Release lock on compilation directory.
    """
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        unlock(get_lock.lock_dir)

def set_lock_status(use_lock):
    """
    Enable or disable the lock on the compilation directory (which is enabled
    by default). Disabling may make compilation slightly faster (but is not
    recommended for parallel execution).

    @param use_lock: whether to use the compilation lock or not
    @type  use_lock: bool
    """
    get_lock.lock_is_enabled = use_lock

def lock(tmp_dir, timeout = 60, min_wait = 5, max_wait = 10, verbosity = 0):
    """
    Obtain lock access by creating a given temporary directory (whose base will
    be created if needed, but will not be deleted after the lock is removed).
    If access is refused by the same lock owner during more than 'timeout'
    seconds, then the current lock is overridden. If timeout is None, then no
    timeout is performed.
    The lock is performed by creating a 'lock' file in 'tmp_dir' that contains
    a unique id identifying the owner of the lock (the process id, followed by
    a random string).
    When there is already a lock, the process sleeps for a random amount of
    time between min_wait and max_wait seconds before trying again.
    If 'verbosity' is >= 1, then a message will be displayed when we need to
    wait for the lock. If it is set to a value >1, then this message will be
    displayed each time we re-check for the presence of the lock. Otherwise it
    is displayed only when we notice the lock's owner has changed.

    @param tmp_dir: lock directory that will be created when acquiring the lock
    @type  tmp_dir: string

    @param timeout: time (in seconds) to wait before replacing an existing lock
    @type  timeout: int

    @param min_wait: minimum time (in seconds) to wait before trying again to
                     get the lock
    @type  min_wait: int

    @param max_wait: maximum time (in seconds) to wait before trying again to
                     get the lock
    @type  max_wait: int

    @param verbosity: amount of feedback displayed to screen
    @type  verbosity: int
    """
    # Create base of lock directory if required.
    base_lock = os.path.basename(tmp_dir)
    if not os.path.isdir(base_lock):
        try:
            os.makedirs(base_lock)
        except:
            # Someone else was probably trying to create it at the same time.
            # We wait two seconds just to make sure the following assert does
            # not fail on some NFS systems.
            os.sleep(2)
    assert os.path.isdir(base_lock)

    # Variable initialization.
    lock_file = os.path.join(tmp_dir, 'lock')
    random.seed()
    unique_id = '%s_%s' % (os.getpid(),
            ''.join([str(random.randint(0,9)) for i in range(10)]))
    no_display = (verbosity == 0)

    # Acquire lock.
    while True:
        try:
            last_owner = 'no_owner'
            time_start = time.time()
            while os.path.isdir(tmp_dir):
                try:
                    read_owner = open(lock_file).readlines()[0].strip()
                except:
                    read_owner = 'failure'
                if last_owner == read_owner:
                    if timeout is not None and time.time() - time_start >= timeout:
                        # Timeout exceeded.
                        break
                else:
                    last_owner = read_owner
                    time_start = time.time()
                    no_display = (verbosity == 0)
                if not no_display:
                    print 'Waiting for existing lock by %s (I am %s)' % (
                            read_owner, unique_id)
                    if verbosity <= 1:
                        no_display = True
                time.sleep(random.uniform(min_wait, max_wait))
    
            try:
                os.mkdir(tmp_dir)
            except:
                # Error while creating the directory: someone else must have tried
                # at the exact same time.
                continue
            # Safety check: the directory should be here.
            assert os.path.isdir(tmp_dir)
    
            # Write own id into lock file.
            lock_write = open(lock_file, 'w')
            lock_write.write(unique_id + '\n')
            lock_write.close()
    
            # Verify we are really the lock owner (this should not be needed,
            # but better be safe than sorry).
            owner = open(lock_file).readlines()[0].strip()
            if owner != unique_id:
                # Too bad, try again.
                continue
            else:
                # We got the lock, hoorray!
                return

        except:
            # If something wrong happened, we try again.
            raise
            continue
    
def unlock(tmp_dir):
    """
    Remove current lock.
    This function assumes we have obtained the lock using lock(tmp_dir, ...),
    so it does not check we are the lock owner.

    @param tmp_dir: lock directory that will be removed when releasing the lock
    @type  tmp_dir: string
    """
    lock_file = os.path.join(tmp_dir, 'lock')
    if os.path.exists(lock_file):
        os.remove(lock_file)
    if os.path.exists(tmp_dir):
        os.rmdir(tmp_dir)
