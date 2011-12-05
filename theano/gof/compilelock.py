# Locking mechanism to ensure no two compilations occur simultaneously in the
# same compilation directory (which can cause crashes).

from theano import config
import compiledir
import os, random, time, atexit
import socket # only used for gethostname()
import logging
_logger=logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.INFO) # INFO will show the the messages "Refreshing lock" message

# In seconds, time that a process will wait before deciding to override an
# existing lock. An override only happens when the existing lock is held by
# the same owner *and* has not been 'refreshed' by this owner for more than
# 'timeout_before_override' seconds.
timeout_before_override = 120

# In seconds, duration before a lock is refreshed. More precisely, the lock is
# refreshed each time 'get_lock()' is called (typically for each file being
# compiled) and the existing lock has not been refreshed in the past
# 'refresh_every' seconds.
refresh_every = 60


def force_unlock():
    """
    Delete the compilation lock if someone else has it.
    """
    global timeout_before_override
    timeout_backup = timeout_before_override
    timeout_before_override = 0
    try:
        get_lock(min_wait=0, max_wait=0.001)
        release_lock()
    finally:
        timeout_before_override = timeout_backup


def get_lock(**kw):
    """
    Obtain lock on compilation directory.

    :param kw: Additional arguments to be forwarded to the `lock` function when
    acquiring the lock.
    """
    if not hasattr(get_lock, 'n_lock'):
        # Initialization.
        get_lock.n_lock = 0
        if not hasattr(get_lock, 'lock_is_enabled'):
            # Enable lock by default.
            get_lock.lock_is_enabled = True
        get_lock.lock_dir = os.path.join(config.compiledir,
                                         'lock_dir')
        get_lock.unlocker = Unlocker(get_lock.lock_dir)
    else:
        lock_dir = os.path.join(config.compiledir, 'lock_dir')
        if lock_dir != get_lock.lock_dir:
            # Compilation directory has changed.
            # First ensure all old locks were released.
            assert get_lock.n_lock == 0
            # Update members for new compilation directory.
            get_lock.lock_dir = lock_dir
            get_lock.unlocker = Unlocker(get_lock.lock_dir)

    if get_lock.lock_is_enabled:
        # Only really try to acquire the lock if we do not have it already.
        if get_lock.n_lock == 0:
            lock(get_lock.lock_dir, timeout=timeout_before_override, **kw)
            atexit.register(Unlocker.unlock, get_lock.unlocker)
            # Store time at which the lock was set.
            get_lock.start_time = time.time()
        else:
            # Check whether we need to 'refresh' the lock. We do this every
            # 'refresh_every' seconds to ensure noone else tries to override
            # our lock after their 'timeout_before_override' timeout period.
            now = time.time()
            if now - get_lock.start_time > refresh_every:
                lockpath = os.path.join(get_lock.lock_dir, 'lock')
                _logger.info('Refreshing lock %s', str(lockpath))
                refresh_lock(lockpath)
                get_lock.start_time = now
    get_lock.n_lock += 1

def release_lock():
    """
    Release lock on compilation directory.
    """
    get_lock.n_lock -= 1
    assert get_lock.n_lock >= 0
    # Only really release lock once all lock requests have ended.
    if get_lock.lock_is_enabled and get_lock.n_lock == 0:
        get_lock.start_time = None
        get_lock.unlocker.unlock()

def set_lock_status(use_lock):
    """
    Enable or disable the lock on the compilation directory (which is enabled
    by default). Disabling may make compilation slightly faster (but is not
    recommended for parallel execution).

    @param use_lock: whether to use the compilation lock or not
    @type  use_lock: bool
    """
    get_lock.lock_is_enabled = use_lock

def lock(tmp_dir, timeout=120, min_wait=5, max_wait=10, verbosity=1):
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
    @type  timeout: int or None

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
    base_lock = os.path.dirname(tmp_dir)
    if not os.path.isdir(base_lock):
        try:
            os.makedirs(base_lock)
        except OSError:
            # Someone else was probably trying to create it at the same time.
            # We wait two seconds just to make sure the following assert does
            # not fail on some NFS systems.
            time.sleep(2)
    assert os.path.isdir(base_lock)

    # Variable initialization.
    lock_file = os.path.join(tmp_dir, 'lock')
    random.seed()
    my_pid = os.getpid()
    no_display = (verbosity == 0)

    # Acquire lock.
    while True:
        try:
            last_owner = 'no_owner'
            time_start = time.time()
            other_dead = False
            while os.path.isdir(tmp_dir):
                try:
                    read_owner = open(lock_file).readlines()[0].strip()
                    # the try is transtion code for old locks
                    # it may be removed when poeple have upgraded
                    try:
                        other_host = read_owner.split('_')[2]
                    except IndexError:
                        other_host = () # make sure it isn't equal to any host
                    if other_host == socket.gethostname():
                        try:
                            os.kill(int(read_owner.split('_')[0]), 0)
                        except OSError:
                            other_dead = True
                        except AttributeError:
                            pass #os.kill does not exist on windows
                except Exception:
                    read_owner = 'failure'
                if other_dead:
                    if not no_display:
                        msg = "process '%s'" % read_owner.split('_')[0]
                        _logger.warning("Overriding existing lock by dead %s "
                                "(I am process '%s')", msg, my_pid)
                    get_lock.unlocker.unlock()
                    continue
                if last_owner == read_owner:
                    if (timeout is not None and
                            time.time() - time_start >= timeout):
                        # Timeout exceeded or locking process dead.
                        if not no_display:
                            if read_owner == 'failure':
                                msg = 'unknown process'
                            else:
                                msg = "process '%s'" % read_owner.split('_')[0]
                            _logger.warning("Overriding existing lock by %s "
                                    "(I am process '%s')", msg, my_pid)
                        get_lock.unlocker.unlock()
                        continue
                else:
                    last_owner = read_owner
                    time_start = time.time()
                    no_display = (verbosity == 0)
                if not no_display:
                    if read_owner == 'failure':
                        msg = 'unknown process'
                    else:
                        msg = "process '%s'" % read_owner.split('_')[0]
                    _logger.info("Waiting for existing lock by %s (I am "
                         "process '%s')", msg, my_pid)
                    _logger.info("To manually release the lock, delete %s", tmp_dir)
                    if verbosity <= 1:
                        no_display = True
                time.sleep(random.uniform(min_wait, max_wait))

            try:
                os.mkdir(tmp_dir)
            except OSError:
                # Error while creating the directory: someone else must have tried
                # at the exact same time.
                continue
            # Safety check: the directory should be here.
            assert os.path.isdir(tmp_dir)

            # Write own id into lock file.
            unique_id = refresh_lock(lock_file)

            # Verify we are really the lock owner (this should not be needed,
            # but better be safe than sorry).
            owner = open(lock_file).readlines()[0].strip()
            if owner != unique_id:
                # Too bad, try again.
                continue
            else:
                # We got the lock, hoorray!
                return

        except Exception, e:
            # If something wrong happened, we try again.
            _logger.warning("Something wrong happened: %s %s", type(e), e)
            time.sleep(random.uniform(min_wait, max_wait))
            continue

def refresh_lock(lock_file):
    """
    'Refresh' an existing lock by re-writing the file containing the owner's
    unique id, using a new (randomly generated) id, which is also returned.
    """
    unique_id = '%s_%s_%s' % (os.getpid(),
            ''.join([str(random.randint(0,9)) for i in range(10)]),
            socket.gethostname())
    lock_write = open(lock_file, 'w')
    lock_write.write(unique_id + '\n')
    lock_write.close()
    return unique_id

class Unlocker(object):
    """
    Class wrapper around release mechanism so that the lock is automatically
    released when the program exits (even when crashing or being interrupted),
    using the __del__ class method.
    """

    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir
        # Keep a pointer to the 'os' module, otherwise it may not be accessible
        # anymore in the __del__ method.
        self.os = os

    def __del__(self):
        self.unlock()

    def unlock(self):
        """
        Remove current lock.
        This function does not crash if it is unable to properly delete the lock
        file and directory. The reason is that it should be allowed for multiple
        jobs running in parallel to unlock the same directory at the same time
        (e.g. when reaching their timeout limit).
        """
        # If any error occurs, we assume this is because someone else tried to
        # unlock this directory at the same time.
        # Note that it is important not to have both remove statements within
        # the same try/except block. The reason is that while the attempt to
        # remove the file may fail (e.g. because for some reason this file does
        # not exist), we still want to try and remove the directory.
        try:
            self.os.remove(self.os.path.join(self.tmp_dir, 'lock'))
        except Exception:
            pass
        try:
            self.os.rmdir(self.tmp_dir)
        except Exception:
            pass
