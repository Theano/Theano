"""Generate and compile C modules for Python,
"""
import atexit
import cPickle
import logging
import operator
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

import distutils.sysconfig

importlib = None
try:
    import importlib
except ImportError:
    pass

import numpy.distutils  # TODO: TensorType should handle this

import theano
from theano.compat import any, PY3, next, decode, decode_iter
from theano.compat.six import BytesIO, StringIO
from theano.gof.utils import flatten
from theano.configparser import config
from theano.gof.cc import hash_from_code
from theano.misc.windows import call_subprocess_Popen

# we will abuse the lockfile mechanism when reading and writing the registry
from theano.gof import compilelock
from theano.gof.compiledir import gcc_version_str, local_bitwidth

from theano.configparser import AddConfigVar, BoolParam

AddConfigVar('cmodule.mac_framework_link',
        "If set to True, breaks certain MacOS installations with the infamous "
        "Bus Error",
        BoolParam(False))

AddConfigVar('cmodule.warn_no_version',
             "If True, will print a warning when compiling one or more Op "
             "with C code that can't be cached because there is no "
             "c_code_cache_version() function associated to at least one of "
             "those Ops.",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('cmodule.remove_gxx_opt',
             "If True, will remove the -O* parameter passed to g++."
             "This is useful to debug in gdb modules compiled by Theano."
             "The parameter -g is passed by default to g++",
             BoolParam(False))

AddConfigVar('cmodule.compilation_warning',
             "If True, will print compilation warnings.",
             BoolParam(False))


_logger = logging.getLogger("theano.gof.cmodule")
_logger.setLevel(logging.WARNING)

METH_VARARGS = "METH_VARARGS"
METH_NOARGS = "METH_NOARGS"


class MissingGXX(Exception):
    """
    This error is raised when we try to generate c code,
    but g++ is not available
    """
    pass


def debug_counter(name, every=1):
    """Debug counter to know how often we go through some piece of code.

    This is a utility function one may use when debugging. Usage example:
        debug_counter('I want to know how often I run this line')
    """
    setattr(debug_counter, name, getattr(debug_counter, name, 0) + 1)
    n = getattr(debug_counter, name)
    if n % every == 0:
        print >>sys.stderr, "debug_counter [%s]: %s" % (name, n)


class ExtFunction(object):
    """A C function to put into a DynamicModule """

    name = ""
    """string - function's name"""

    code_block = ""
    """string - the entire code for the function.

    Has the form ``static PyObject* <name>([...]){ ... }

    See Python's C API Reference for how to write c functions for python
    modules.
    """

    method = ""
    """
    str - calling method for this function (i.e. 'METH_VARARGS', 'METH_NOARGS')
    """

    doc = ""
    """str - documentation string for this function"""

    def __init__(self, name, code_block, method, doc="undocumented"):
        self.name = name
        self.code_block = code_block
        self.method = method
        self.doc = doc

    def method_decl(self):
        """
        Returns the signature for this function.

        It goes into the DynamicModule's method table.
        """
        return '\t{"%s", %s, %s, "%s"}' % (
                self.name, self.name, self.method, self.doc)


class DynamicModule(object):
    def __init__(self, name=None):
        assert name is None, ("The 'name' parameter of DynamicModule"
                " cannot be specified anymore. Instead, 'code_hash'"
                " will be automatically computed and can be used as"
                " the module's name.")
        # While the module is not finalized, we can call add_...
        # when it is finalized, a hash is computed and used instead of
        # the placeholder, and as module name.
        self.finalized = False
        self.code_hash = None
        self.hash_placeholder = '<<<<HASH_PLACEHOLDER>>>>'

        self.support_code = []
        self.functions = []
        self.includes = ["<Python.h>", "<iostream>"]
        self.init_blocks = []

    def print_methoddef(self, stream):
        print >> stream, "static PyMethodDef MyMethods[] = {"
        for f in self.functions:
            print >> stream, f.method_decl(), ','
        print >> stream, "\t{NULL, NULL, 0, NULL}"
        print >> stream, "};"

    def print_init(self, stream):
        if PY3:
            print >> stream, """\
static struct PyModuleDef moduledef = {{
      PyModuleDef_HEAD_INIT,
      "{name}",
      NULL,
      -1,
      MyMethods,
}};
""".format(name=self.hash_placeholder)
            print >> stream, ("PyMODINIT_FUNC PyInit_%s(void) {" %
                              self.hash_placeholder)
            for block in self.init_blocks:
                print >> stream, '  ', block
            print >> stream, "    PyObject *m = PyModule_Create(&moduledef);"
            print >> stream, "    return m;"
        else:
            print >> stream, ("PyMODINIT_FUNC init%s(void){" %
                              self.hash_placeholder)
            for block in self.init_blocks:
                print >> stream, '  ', block
            print >> stream, '  ', ('(void) Py_InitModule("%s", MyMethods);'
                                    % self.hash_placeholder)
        print >> stream, "}"

    def add_include(self, str):
        assert not self.finalized
        self.includes.append(str)

    def add_init_code(self, code):
        assert not self.finalized
        self.init_blocks.append(code)

    def add_support_code(self, code):
        assert not self.finalized
        if code not in self.support_code:  # TODO: KLUDGE
            self.support_code.append(code)

    def add_function(self, fn):
        assert not self.finalized
        self.functions.append(fn)

    def code(self):
        sio = StringIO()
        for inc in self.includes:
            if not inc:
                continue
            if inc[0] == '<' or inc[0] == '"':
                print >> sio, "#include", inc
            else:
                print >> sio, '#include "%s"' % inc

        print >> sio, "//////////////////////"
        print >> sio, "////  Support Code"
        print >> sio, "//////////////////////"
        for sc in self.support_code:
            print >> sio, sc

        print >> sio, "//////////////////////"
        print >> sio, "////  Functions"
        print >> sio, "//////////////////////"
        for f in self.functions:
            print >> sio, f.code_block

        print >> sio, "//////////////////////"
        print >> sio, "////  Module init"
        print >> sio, "//////////////////////"
        self.print_methoddef(sio)
        self.print_init(sio)

        rval = sio.getvalue()
        self.code_hash = hash_from_code(rval)
        rval = re.sub(self.hash_placeholder, self.code_hash, rval)
        # Finalize the Module, so no support code or function
        # can be added
        self.finalized = True

        return rval

    def list_code(self, ofile=sys.stdout):
        """Print out the code with line numbers to `ofile` """
        for i, line in enumerate(self.code().split('\n')):
            print >> ofile, ('%4i' % (i + 1)), line
        ofile.flush()

    #TODO: add_type


def dlimport(fullpath, suffix=None):
    """Dynamically load a .so, .pyd, .dll, or .py file

    :type fullpath: string
    :param fullpath: a fully-qualified path do a compiled python module
    :param suffix: a suffix to strip from the end of fullpath to get the
        import name
    :type suffix: string

    :returns: the dynamically loaded module (from __import__)

    """
    if not os.path.isabs(fullpath):
        raise ValueError('`fullpath` must be an absolute path', fullpath)
    if suffix is None:
        if fullpath.endswith('.so'):
            suffix = '.so'
        elif fullpath.endswith('.pyd'):
            suffix = '.pyd'
        elif fullpath.endswith('.dll'):
            suffix = '.dll'
        elif fullpath.endswith('.py'):
            suffix = '.py'
        else:
            suffix = ''
    rval = None
    if fullpath.endswith(suffix):
        module_name = '.'.join(fullpath.split(os.path.sep)[-2:])[:-len(suffix)]
    else:
        raise ValueError('path has wrong suffix', (fullpath, suffix))
    workdir = fullpath[:-len(module_name) - 1 - len(suffix)]

    _logger.debug("WORKDIR %s", workdir)
    _logger.debug("module_name %s", module_name)

    sys.path[0:0] = [workdir]  # insert workdir at beginning (temporarily)
    try:
        if importlib is not None:
            if hasattr(importlib, "invalidate_caches"):
                importlib.invalidate_caches()
        rval = __import__(module_name, {}, {}, [module_name])
        if not rval:
            raise Exception('__import__ failed', fullpath)
    finally:
        del sys.path[0]

    assert fullpath.startswith(rval.__file__)
    return rval


def dlimport_workdir(basedir):
    """
    Return a directory where you should put your .so file for dlimport
    to be able to load it, given a basedir which should normally be
    config.compiledir
    """
    return tempfile.mkdtemp(dir=basedir)


def last_access_time(path):
    """
    Return the number of seconds since the epoch of the last access of a
    given file.
    """
    return os.stat(path)[stat.ST_ATIME]


def module_name_from_dir(dirname, err=True):
    """
    Scan the contents of a cache directory and return full path of the
    dynamic lib in it.
    """
    files = os.listdir(dirname)
    names = [file for file in files
             if file.endswith('.so') or file.endswith('.pyd')]
    if len(names) == 0 and not err:
        return None
    elif len(names) == 1:
        return os.path.join(dirname, names[0])
    else:
        raise ValueError("More than 1 compiled module in this directory:" +
                         dirname)


def is_same_entry(entry_1, entry_2):
    """
    Return True iff both paths can be considered to point to the same module.

    This is the case if and only if at least one of these conditions holds:
        - They are equal.
        - Their real paths are equal.
        - They share the same temporary work directory and module file name.
    """
    if entry_1 == entry_2:
        return True
    if os.path.realpath(entry_1) == os.path.realpath(entry_2):
        return True
    if (os.path.basename(entry_1) == os.path.basename(entry_2) and
        (os.path.basename(os.path.dirname(entry_1)) ==
         os.path.basename(os.path.dirname(entry_2))) and
        os.path.basename(os.path.dirname(entry_1)).startswith('tmp')):
        return True
    return False


def get_module_hash(src_code, key):

    """
    Return an MD5 hash that uniquely identifies a module.

    This hash takes into account:
        1. The C source code of the module (`src_code`).
        2. The version part of the key.
        3. The compiler options defined in `key` (command line parameters and
           libraries to link against).
        4. The NumPy ABI version.
    """
    # `to_hash` will contain any element such that we know for sure that if
    # it changes, then the module hash should be different.
    # We start with the source code itself (stripping blanks might avoid
    # recompiling after a basic indentation fix for instance).
    to_hash = map(str.strip, src_code.split('\n'))
    # Get the version part of the key (ignore if unversioned).
    if key[0]:
        to_hash += map(str, key[0])
    c_link_key = key[1]
    # Currently, in order to catch potential bugs early, we are very
    # convervative about the structure of the key and raise an exception
    # if it does not match exactly what we expect. In the future we may
    # modify this behavior to be less strict and be able to accomodate
    # changes to the key in an automatic way.
    # Note that if the key structure changes, the `get_safe_part` fucntion
    # below may also need to be modified.
    error_msg = ("This should not happen unless someone modified the code "
                 "that defines the CLinker key, in which case you should "
                 "ensure this piece of code is still valid (and this "
                 "AssertionError may be removed or modified to accomodate "
                 "this change)")
    assert c_link_key[0] == 'CLinker.cmodule_key', error_msg
    for key_element in c_link_key[1:]:
        if isinstance(key_element, tuple):
            # This should be the C++ compilation command line parameters or the
            # libraries to link against.
            to_hash += list(key_element)
        elif isinstance(key_element, basestring):
            if key_element.startswith('md5:'):
                # This is the md5 hash of the config options. We can stop
                # here.
                break
            elif (key_element.startswith('NPY_ABI_VERSION=0x') or
                  key_element.startswith('c_compiler_str=')):
                to_hash.append(key_element)
            else:
                raise AssertionError(error_msg)
        else:
            raise AssertionError(error_msg)
    return hash_from_code('\n'.join(to_hash))


def get_safe_part(key):
    """
    Return a tuple containing a subset of `key`, to be used to find equal keys.

    This tuple should only contain objects whose __eq__ and __hash__ methods
    can be trusted (currently: the version part of the key, as well as the
    md5 hash of the config options).
    It is used to reduce the amount of key comparisons one has to go through
    in order to find broken keys (i.e. keys with bad implementations of __eq__
    or __hash__).
    """
    version = key[0]
    # This function should only be called on versioned keys.
    assert version

    # Find the md5 hash part.
    c_link_key = key[1]
    for key_element in c_link_key[1:]:
        if (isinstance(key_element, basestring)
                and key_element.startswith('md5:')):
            md5 = key_element[4:]
            break

    return key[0] + (md5, )


class KeyData(object):

    """Used to store the key information in the cache."""

    def __init__(self, keys, module_hash, key_pkl, entry):
        """
        Constructor.

        :param keys: Set of keys that are associated to the exact same module.

        :param module_hash: Hash identifying the module (it should hash both
        the code and the compilation options).

        :param key_pkl: Path to the file in which this KeyData object should be
        pickled.
        """
        self.keys = keys
        self.module_hash = module_hash
        self.key_pkl = key_pkl
        self.entry = entry

    def add_key(self, key, save_pkl=True):
        """Add a key to self.keys, and update pickled file if asked to."""
        assert key not in self.keys
        self.keys.add(key)
        if save_pkl:
            self.save_pkl()

    def remove_key(self, key, save_pkl=True):
        """Remove a key from self.keys, and update pickled file if asked to."""
        self.keys.remove(key)
        if save_pkl:
            self.save_pkl()

    def save_pkl(self):
        """
        Dump this object into its `key_pkl` file.

        May raise a cPickle.PicklingError if such an exception is raised at
        pickle time (in which case a warning is also displayed).
        """
        # Note that writing in binary mode is important under Windows.
        try:
            cPickle.dump(self, open(self.key_pkl, 'wb'),
                         protocol=cPickle.HIGHEST_PROTOCOL)
        except cPickle.PicklingError:
            _logger.warning("Cache leak due to unpickle-able key data %s",
                            self.keys)
            os.remove(self.key_pkl)
            raise

    def get_entry(self):
        """Return path to the module file."""
        # TODO This method may be removed in the future (e.g. in 0.5) since
        # its only purpose is to make sure that old KeyData objects created
        # before the 'entry' field was added are properly handled.
        if not hasattr(self, 'entry'):
            self.entry = module_name_from_dir(os.path.dirname(self.key_pkl))
        return self.entry

    def delete_keys_from(self, entry_from_key, do_manual_check=True):
        """
        Delete from entry_from_key all keys associated to this KeyData object.

        Note that broken keys will not appear in the keys field, so we also
        manually look for keys associated to the same entry, unless
        do_manual_check is False.
        """
        entry = self.get_entry()
        for key in self.keys:
            del entry_from_key[key]
        if do_manual_check:
            to_del = []
            for key, key_entry in entry_from_key.iteritems():
                if key_entry == entry:
                    to_del.append(key)
            for key in to_del:
                del entry_from_key[key]


class ModuleCache(object):
    """Interface to the cache of dynamically compiled modules on disk

    Note that this interface does not assume exclusive use of the cache
    directory. It is built to handle the case where multiple programs are also
    using instances of this class to manage the same directory.

    The cache works on the basis of keys. Each key is mapped to only one
    dynamic module, but multiple keys may be mapped to the same module (see
    below for details). Each module is a dynamic library file, that Python
    can import.

    The cache contains one directory for each module, containing:
    - the dynamic library file itself (.so/.pyd),
    - an empty __init__.py file, so Python can import it,
    - a file containing the source code for the module (mod.cpp/mod.cu),
    - a key.pkl file, containing a KeyData object with all the keys
    associated with that module,
    - possibly a delete.me file, meaning this directory has been marked
    for deletion.

    Keys should be tuples of length 2: (version, rest). The
    ``rest`` can be anything hashable and picklable, that uniquely
    identifies the computation in the module. The key is returned by
    ``CLinker.cmodule_key_``.

    The ``version`` should be a hierarchy of tuples of integers.
    If the ``version`` is either 0 or (), then the key is unversioned, and its
    corresponding module will be marked for deletion in an atexit() handler.
    If the ``version`` is neither 0 nor (), then the module will be kept in the
    cache between processes.

    An unversioned module is not always deleted by the process that
    creates it.  Deleting such modules may not work on NFS filesystems
    because the tmpdir in which the library resides is in use until the
    end of the process' lifetime.  In this case, unversioned modules
    are left in their tmpdirs without corresponding .pkl files.  These
    modules and their directories are erased by subsequent processes'
    refresh() functions.

    Two different keys are mapped to the same module when all conditions below
    are met:
        - They have the same version.
        - They share the same compilation options in their ``rest`` part (see
          ``CLinker.cmodule_key_`` for how this part is built).
        - They share the same C code.
    These three elements uniquely identify a module, and are summarized
    in a single "module hash".
    """

    dirname = ""
    """The working directory that is managed by this interface"""

    module_from_name = {}
    """maps a module filename to the loaded module object"""

    entry_from_key = {}
    """Maps keys to the filename of a .so/.pyd.
    """

    similar_keys = {}
    """Maps a part-of-key to all keys that share this same part."""

    module_hash_to_key_data = {}
    """Maps a module hash to its corresponding KeyData object."""

    stats = []
    """
    A list with counters for the number of hits, loads, compiles issued by
    module_from_key()
    """

    loaded_key_pkl = set()
    """set of all key.pkl files that have been loaded.
    """

    def __init__(self, dirname, check_for_broken_eq=True, do_refresh=True):
        """
        :param check_for_broken_eq: A bad __eq__ implementation can break this
        cache mechanism. This option turns on a not-too-expensive sanity check
        every time a new key is added to the cache.

        :param do_refresh: If True, then the ``refresh`` method will be called
        in the constructor.
        """
        self.dirname = dirname
        self.module_from_name = dict(self.module_from_name)
        self.entry_from_key = dict(self.entry_from_key)
        self.module_hash_to_key_data = dict(self.module_hash_to_key_data)
        self.similar_keys = dict(self.similar_keys)
        self.stats = [0, 0, 0]
        self.check_for_broken_eq = check_for_broken_eq
        self.loaded_key_pkl = set()
        self.time_spent_in_check_key = 0

        if do_refresh:
            self.refresh()

    age_thresh_use = 60 * 60 * 24 * 24    # 24 days
    """
    The default age threshold (in seconds) for cache files we want to use.

    Older modules will be deleted in ``clear_old``.
    """

    def refresh(self, age_thresh_use=None, delete_if_problem=False):
        """Update cache data by walking the cache directory structure.

        Load key.pkl files that have not been loaded yet.
        Remove entries which have been removed from the filesystem.
        Also, remove malformed cache directories.

        :param age_thresh_use: Do not use modules olther than this.
        Defaults to self.age_thresh_use.

        :param delete_if_problem: If True, cache entries that meet one of those
        two conditions are deleted:
            - Those for which unpickling the KeyData file fails with an
              unknown exception.
            - Duplicated modules, regardless of their age.

        :returns: a list of modules of age higher than age_thresh_use.
        """
        if age_thresh_use is None:
            age_thresh_use = self.age_thresh_use
        start_time = time.time()
        too_old_to_use = []

        compilelock.get_lock()
        try:
            # add entries that are not in the entry_from_key dictionary
            time_now = time.time()
            # Go through directories in alphabetical order to ensure consistent
            # behavior.
            root_dirs_files = sorted(os.walk(self.dirname),
                                     key=operator.itemgetter(0))
            for root, dirs, files in root_dirs_files:
                key_pkl = os.path.join(root, 'key.pkl')
                if key_pkl in self.loaded_key_pkl:
                    continue
                elif 'delete.me' in files or not files:
                    _rmtree(root, ignore_nocleanup=True,
                            msg="delete.me found in dir")
                elif 'key.pkl' in files:
                    try:
                        entry = module_name_from_dir(root)
                    except ValueError:  # there is a key but no dll!
                        if not root.startswith("/tmp"):
                            # Under /tmp, file are removed periodically by the
                            # os. So it is normal that this happens from time
                            # to time.
                            _logger.warning("ModuleCache.refresh() Found key "
                                            "without dll in cache, deleting it. %s",
                                            key_pkl)
                        _rmtree(root, ignore_nocleanup=True,
                                msg="missing module file", level=logging.INFO)
                        continue
                    if (time_now - last_access_time(entry)) < age_thresh_use:
                        _logger.debug('refresh adding %s', key_pkl)

                        def unpickle_failure():
                            _logger.info("ModuleCache.refresh() Failed to "
                                         "unpickle cache file %s", key_pkl)

                        try:
                            key_data = cPickle.load(open(key_pkl, 'rb'))
                        except EOFError:
                            # Happened once... not sure why (would be worth
                            # investigating if it ever happens again).
                            unpickle_failure()
                            _rmtree(root, ignore_nocleanup=True,
                                    msg='broken cache directory [EOF]',
                                    level=logging.WARNING)
                            continue
                        except ValueError:
                            # This can happen when we have bad config value
                            # in the cuda.nvcc_compiler.py file.
                            # We should not hide it here, as this will cause
                            # an unrelated error to appear.
                            raise
                        except Exception:
                            unpickle_failure()
                            if delete_if_problem:
                                _rmtree(root, ignore_nocleanup=True,
                                        msg='broken cache directory',
                                        level=logging.INFO)
                            else:
                                # This exception is often triggered by keys
                                # that contain references to classes that have
                                # not yet been imported (e.g. when running two
                                # different Theano-based scripts). They are not
                                # necessarily broken, but we cannot load them
                                # here.
                                pass
                            continue

                        if not isinstance(key_data, KeyData):
                            # This is some old cache data, that does not fit
                            # the new cache format. It would be possible to
                            # update it, but it is not entirely safe since we
                            # do not know the config options that were used.
                            # As a result, we delete it instead (which is also
                            # simpler to implement).
                            _rmtree(root, ignore_nocleanup=True,
                                    msg=(
                                        'invalid cache entry format -- this '
                                        'should not happen unless your cache '
                                        'was really old'),
                                    level=logging.WARN)
                            continue

                        # Check the path to the module stored in the KeyData
                        # object matches the path to `entry`. There may be
                        # a mismatch e.g. due to symlinks, or some directory
                        # being renamed since last time cache was created.
                        kd_entry = key_data.get_entry()
                        if kd_entry != entry:
                            if is_same_entry(entry, kd_entry):
                                # Update KeyData object. Note that we also need
                                # to update the key_pkl field, because it is
                                # likely to be incorrect if the entry itself
                                # was wrong.
                                key_data.entry = entry
                                key_data.key_pkl = key_pkl
                            else:
                                # This is suspicious. Better get rid of it.
                                _rmtree(root, ignore_nocleanup=True,
                                        msg='module file path mismatch',
                                        level=logging.INFO)
                                continue

                        # Find unversioned keys from other processes.
                        # TODO: check if this can happen at all
                        to_del = [key for key in key_data.keys if not key[0]]
                        if to_del:
                            _logger.warning(
                                    "ModuleCache.refresh() Found unversioned "
                                    "key in cache, removing it. %s", key_pkl)
                            # Since the version is in the module hash, all
                            # keys should be unversioned.
                            if len(to_del) != len(key_data.keys):
                                _logger.warning(
                                        'Found a mix of unversioned and '
                                        'versioned keys for the same '
                                        'module %s', key_pkl)
                            _rmtree(root, ignore_nocleanup=True,
                                    msg="unversioned key(s) in cache",
                                    level=logging.INFO)
                            continue

                        mod_hash = key_data.module_hash
                        if mod_hash in self.module_hash_to_key_data:
                            # This may happen when two processes running
                            # simultaneously compiled the same module, one
                            # after the other. We delete one once it is old
                            # enough (to be confident there is no other process
                            # using it), or if `delete_if_problem` is True.
                            # Note that it is important to walk through
                            # directories in alphabetical order so as to make
                            # sure all new processes only use the first one.
                            age = time.time() - last_access_time(entry)
                            if delete_if_problem or age > self.age_thresh_del:
                                _rmtree(root, ignore_nocleanup=True,
                                        msg='duplicated module',
                                        level=logging.DEBUG)
                            else:
                                _logger.debug('Found duplicated module not '
                                              'old enough yet to be deleted '
                                              '(age: %s): %s',
                                              age, entry)
                            continue

                        # Remember the map from a module's hash to the KeyData
                        # object associated with it.
                        self.module_hash_to_key_data[mod_hash] = key_data

                        for key in key_data.keys:
                            if key not in self.entry_from_key:
                                self.entry_from_key[key] = entry
                                # Assert that we have not already got this
                                # entry somehow.
                                assert entry not in self.module_from_name
                                # Store safe part of versioned keys.
                                if key[0]:
                                    self.similar_keys.setdefault(
                                            get_safe_part(key),
                                            []).append(key)
                            else:
                                _logger.warning(
                                    "The same cache key is associated to "
                                    "different modules (%s and %s). This "
                                    "is not supposed to happen! You may "
                                    "need to manually delete your cache "
                                    "directory to fix this.",
                                    self.entry_from_key[key],
                                    entry)
                        # Clean up the name space to prevent bug.
                        if key_data.keys:
                            del key
                        self.loaded_key_pkl.add(key_pkl)
                    else:
                        too_old_to_use.append(entry)

                # If the compilation failed, no key.pkl is in that
                # directory, but a mod.* should be there.
                # We do nothing here.

            # Clean up the name space to prevent bug.
            if root_dirs_files:
                del root, dirs, files

            # Remove entries that are not in the filesystem.
            items_copy = list(self.module_hash_to_key_data.iteritems())
            for module_hash, key_data in items_copy:
                entry = key_data.get_entry()
                try:
                    # Test to see that the file is [present and] readable.
                    open(entry).close()
                    gone = False
                except IOError:
                    gone = True

                if gone:
                    # Assert that we did not have one of the deleted files
                    # loaded up and in use.
                    # If so, it should not have been deleted. This should be
                    # considered a failure of the OTHER process, that deleted
                    # it.
                    if entry in self.module_from_name:
                        _logger.warning("A module that was loaded by this "
                                "ModuleCache can no longer be read from file "
                                "%s... this could lead to problems.",
                                entry)
                        del self.module_from_name[entry]

                    _logger.info("deleting ModuleCache entry %s", entry)
                    key_data.delete_keys_from(self.entry_from_key)
                    del self.module_hash_to_key_data[module_hash]
                    if key_data.keys and list(key_data.keys)[0][0]:
                        # this is a versioned entry, so should have been on
                        # disk. Something weird happened to cause this, so we
                        # are responding by printing a warning, removing
                        # evidence that we ever saw this mystery key.
                        pkl_file_to_remove = key_data.key_pkl
                        if not key_data.key_pkl.startswith("/tmp"):
                            # Under /tmp, file are removed periodically by the
                            # os. So it is normal that this happen from time to
                            # time.
                            _logger.warning("Removing key file %s because the "
                                    "corresponding module is gone from the "
                                    "file system.",
                                    pkl_file_to_remove)
                        self.loaded_key_pkl.remove(pkl_file_to_remove)

        finally:
            compilelock.release_lock()

        _logger.debug('Time needed to refresh cache: %s',
                      (time.time() - start_time))

        return too_old_to_use

    def module_from_key(self, key, fn=None, keep_lock=False, key_data=None):
        """
        :param fn: A callable object that will return an iterable object when
        called, such that the first element in this iterable object is the
        source code of the module, and the last element is the module itself.
        `fn` is called only if the key is not already in the cache, with
        a single keyword argument `location` that is the path to the directory
        where the module should be compiled.

        :param key_data: If not None, it should be a KeyData object and the
        key parameter should be None. In this case, we use the info from the
        KeyData object to recover the module, rather than the key itself. Note
        that this implies the module already exists (and may or may not have
        already been loaded).
        """
        # We should only use one of the two ways to get a module.
        assert key_data is None or key is None
        rval = None
        if key is not None:
            try:
                _version, _rest = key
            except (TypeError, ValueError):
                raise ValueError(
                        "Invalid key. key must have form (version, rest)", key)
        name = None
        if key is not None and key in self.entry_from_key:
            # We have seen this key either in this process or previously.
            name = self.entry_from_key[key]
        elif key_data is not None:
            name = key_data.get_entry()
        if name is not None:
            # This is an existing module we can recover.
            if name not in self.module_from_name:
                _logger.debug('loading name %s', name)
                self.module_from_name[name] = dlimport(name)
                self.stats[1] += 1
            else:
                self.stats[0] += 1
            _logger.debug('returning compiled module from cache %s', name)
            rval = self.module_from_name[name]
        else:
            hash_key = hash(key)
            key_data = None
            # We have never seen this key before.

            # We acquire the lock later only if we were able to
            # generate C code. Otherwise, we would take the lock for ops
            # that have only a perform().
            lock_taken = False
            # This try/finally block ensures that the lock is released once we
            # are done writing in the cache file or after raising an exception.
            try:
                # Embedding two try statements for Python 2.4 compatibility
                # (cannot do try / except / finally).
                try:
                    location = dlimport_workdir(self.dirname)
                except OSError, e:
                    _logger.error(e)
                    if e.errno == 31:
                        _logger.error('There are %i files in %s',
                                      len(os.listdir(config.compiledir)),
                                      config.compiledir)
                    raise
                try:
                    compile_steps = fn(location=location).__iter__()

                    # Check if we already know a module with the same hash.
                    # If we do, then there is no need to even compile it.
                    duplicated_module = False
                    # The first compilation step is to yield the source code.
                    src_code = next(compile_steps)
                    module_hash = get_module_hash(src_code, key)

                    # The op has c_code, so take the lock.
                    compilelock.get_lock()
                    lock_taken = True

                    if not os.path.exists(location):
                        # Temporary fix, we should make sure it don't
                        # get deleted by the clear*() fct.
                        os.makedirs(location)

                    if module_hash in self.module_hash_to_key_data:
                        _logger.debug("Duplicated module! Will re-use the "
                                      "previous one")
                        duplicated_module = True
                        # Load the already existing module.
                        key_data = self.module_hash_to_key_data[module_hash]
                        # Note that we do not pass the `fn` argument, since it
                        # should not be used considering that the module should
                        # already be compiled.
                        module = self.module_from_key(key=None,
                                                      key_data=key_data)
                        name = module.__file__
                        # Add current key to the set of keys associated to the
                        # same module. We only save the KeyData object of
                        # versioned modules.
                        try:
                            key_data.add_key(key, save_pkl=bool(_version))
                            key_broken = False
                        except cPickle.PicklingError:
                            # This should only happen if we tried to save the
                            # pickled file.
                            assert _version
                            # The key we are trying to add is broken: we will
                            # not add it after all.
                            key_data.remove_key(key)
                            key_broken = True

                        if (_version and not key_broken and
                            self.check_for_broken_eq):
                            self.check_key(key, key_data.key_pkl)

                        # We can delete the work directory.
                        _rmtree(location, ignore_nocleanup=True,
                                msg='temporary workdir of duplicated module')

                    else:
                        # Will fail if there is an error compiling the C code.
                        # The exception will be caught and the work dir will be
                        # deleted.
                        while True:
                            try:
                                # The module should be returned by the last
                                # step of the compilation.
                                module = next(compile_steps)
                            except StopIteration:
                                break

                        # Obtain path to the '.so' module file.
                        name = module.__file__

                        _logger.debug("Adding module to cache %s %s",
                                      key, name)
                        assert name.startswith(location)
                        assert name not in self.module_from_name
                        # Changing the hash of the key is not allowed during
                        # compilation. That is the only cause found that makes
                        # the following assert fail.
                        assert hash(key) == hash_key
                        assert key not in self.entry_from_key

                        key_pkl = os.path.join(location, 'key.pkl')
                        assert not os.path.exists(key_pkl)
                        key_data = KeyData(
                                keys=set([key]),
                                module_hash=module_hash,
                                key_pkl=key_pkl,
                                entry=name)

                        # Note that we only save KeyData objects associated to
                        # versioned modules. So for unversioned key, the
                        # `key_pkl` field of the KeyData object will be a
                        # non-existing file (which does not matter since it
                        # will not be accessed).
                        if _version:
                            try:
                                key_data.save_pkl()
                                key_broken = False
                            except cPickle.PicklingError:
                                key_broken = True
                                # Remove key from the KeyData object, to make
                                # sure we never try to save it again.
                                # We still keep the KeyData object and save it
                                # so that the module can be re-used in the
                                # future.
                                key_data.keys = set()
                                key_data.save_pkl()

                            if not key_broken and self.check_for_broken_eq:
                                self.check_key(key, key_pkl)

                            # Adding the KeyData file to this set means it is a
                            # versioned module.
                            self.loaded_key_pkl.add(key_pkl)
                        elif config.cmodule.warn_no_version:
                            key_flat = flatten(key)
                            ops = [k for k in key_flat
                                   if isinstance(k, theano.Op)]
                            _logger.warning("not all the"
                                " following op(s) implement"
                                " c_code_cache_version(). This makes them"
                                " recompiled for each process." + str(ops))

                        # Map the new module to its KeyData object. Note that
                        # we need to do it regardless of whether the key is
                        # versioned or not if we want to be able to re-use this
                        # module inside the same process.
                        self.module_hash_to_key_data[module_hash] = key_data

                except Exception:
                    # This may happen e.g. when an Op has no C implementation.
                    # In any case, we do not want to keep around the temporary
                    # work directory, as it may cause trouble if we create too
                    # many of these. The 'ignore_if_missing' flag is set just
                    # in case this directory would have already been deleted.
                    _rmtree(location, ignore_if_missing=True,
                            msg=('exception -- '
                                 'typically means no C implementation'))
                    raise

            finally:
                # Release lock if needed.
                if not keep_lock and lock_taken:
                    compilelock.release_lock()

            # Update map from key to module name for all keys associated to
            # this same module.
            all_keys = key_data.keys
            if not all_keys:
                # Should only happen for broken keys.
                assert key_broken
                all_keys = [key]
            else:
                assert key in key_data.keys
            for k in all_keys:
                if k in self.entry_from_key:
                    # If we had already seen this key, then it should be
                    # associated to the same module.
                    assert self.entry_from_key[k] == name
                else:
                    self.entry_from_key[k] = name
                    if _version:
                        self.similar_keys.setdefault(get_safe_part(k),
                                                     []).append(key)

            if name in self.module_from_name:
                # May happen if we are re-using an existing module.
                assert duplicated_module
                assert self.module_from_name[name] is module
            else:
                self.module_from_name[name] = module

            self.stats[2] += 1
            rval = module
        #_logger.debug('stats %s %i', self.stats, sum(self.stats))
        return rval

    def check_key(self, key, key_pkl):
        """
        Perform checks to detect broken __eq__ / __hash__ implementations.

        :param key: The key to be checked.
        :param key_pkl: Its associated pickled file containing a KeyData.
        """
        start_time = time.time()
        # Verify that when we reload the KeyData from the pickled file, the
        # same key can be found in it, and is not equal to more than one
        # other key.
        key_data = cPickle.load(open(key_pkl, 'rb'))
        found = sum(key == other_key for other_key in key_data.keys)
        msg = ''
        if found == 0:
            msg = 'Key not found in unpickled KeyData file'
            if key_data.keys:
                # This is to make debugging in pdb easier, by providing
                # the offending keys in the local context.
                # key_data_keys = list(key_data.keys)
                ## import pdb; pdb.set_trace()
                pass
        elif found > 1:
            msg = 'Multiple equal keys found in unpickled KeyData file'
        if msg:
            raise AssertionError(
                    "%s. Verify the __eq__ and __hash__ functions of your "
                    "Ops. The file is: %s. The key is: %s" %
                    (msg, key_pkl, key))
        # Also verify that there exists no other loaded key that would be equal
        # to this key. In order to speed things up, we only compare to keys
        # with the same version part and config md5, since we can assume this
        # part of the key is not broken.
        for other in self.similar_keys.get(get_safe_part(key), []):
            if other is not key and other == key and hash(other) != hash(key):
                raise AssertionError(
                    "Found two keys that are equal but have a different hash. "
                    "Verify the __eq__ and __hash__ functions of your Ops. "
                    "The keys are:\n  %s\nand\n  %s\n(found in %s)." %
                    (other, key, key_pkl))

        self.time_spent_in_check_key += time.time() - start_time

    age_thresh_del = 60 * 60 * 24 * 31  # 31 days
    age_thresh_del_unversioned = 60 * 60 * 24 * 7  # 7 days

    """The default age threshold for `clear_old` (in seconds)
    """
    def clear_old(self, age_thresh_del=None, delete_if_problem=False):
        """
        Delete entries from the filesystem for cache entries that are too old.

        :param age_thresh_del: Dynamic modules whose last access time is more
        than ``age_thresh_del`` seconds ago will be erased. Defaults to 31-day
        age if not provided.

        :param delete_if_problem: See help of refresh() method.
        """
        if age_thresh_del is None:
            age_thresh_del = self.age_thresh_del

        # Ensure that the too_old_to_use list return by refresh() will
        # contain all modules older than age_thresh_del.
        if age_thresh_del < self.age_thresh_use:
            if age_thresh_del > 0:
                _logger.warning("Clearing modules that were not deemed "
                        "too old to use: age_thresh_del=%d, "
                        "self.age_thresh_use=%d",
                        age_thresh_del,
                        self.age_thresh_use)
            else:
                _logger.info("Clearing all modules.")
            age_thresh_use = age_thresh_del
        else:
            age_thresh_use = None

        compilelock.get_lock()
        try:
            # Update the age of modules that have been accessed by other
            # processes and get all module that are too old to use
            # (not loaded in self.entry_from_key).
            too_old_to_use = self.refresh(
                    age_thresh_use=age_thresh_use,
                    delete_if_problem=delete_if_problem)

            for entry in too_old_to_use:
                # TODO: we are assuming that modules that haven't been
                # accessed in over age_thresh_del are not currently in
                # use by other processes, but that could be false for
                # long-running jobs, or if age_thresh_del < 0.
                assert entry not in self.module_from_name
                parent = os.path.dirname(entry)
                assert parent.startswith(os.path.join(self.dirname, 'tmp'))
                _rmtree(parent, msg='old cache directory', level=logging.INFO,
                        ignore_nocleanup=True)

        finally:
            compilelock.release_lock()

    def clear(self, unversioned_min_age=None, clear_base_files=False,
              delete_if_problem=False):
        """
        Clear all elements in the cache.

        :param unversioned_min_age: Forwarded to `clear_unversioned`. In
        particular, you can set it to -1 in order to delete all unversioned
        cached modules regardless of their age.

        :param clear_base_files: If True, then delete base directories
        'cuda_ndarray', 'cutils_ext', 'lazylinker_ext' and 'scan_perform'
        if they are present.
        If False, those directories are left intact.

        :param delete_if_problem: See help of refresh() method.
        """
        compilelock.get_lock()
        try:
            self.clear_old(
                    age_thresh_del=-1.0,
                    delete_if_problem=delete_if_problem)
            self.clear_unversioned(min_age=unversioned_min_age)
            if clear_base_files:
                self.clear_base_files()
        finally:
            compilelock.release_lock()

    def clear_base_files(self):
        """
        Remove base directories 'cuda_ndarray', 'cutils_ext', 'lazylinker_ext'
        and 'scan_perform' if present.

        Note that we do not delete them outright because it may not work on
        some systems due to these modules being currently in use. Instead we
        rename them with the '.delete.me' extension, to mark them to be deleted
        next time we clear the cache.
        """
        compilelock.get_lock()
        try:
            for base_dir in ('cuda_ndarray', 'cutils_ext', 'lazylinker_ext',
                             'scan_perform'):
                to_delete = os.path.join(self.dirname, base_dir + '.delete.me')
                if os.path.isdir(to_delete):
                    try:
                        shutil.rmtree(to_delete)
                        _logger.debug('Deleted: %s', to_delete)
                    except Exception:
                        _logger.warning('Could not delete %s', to_delete)
                        continue
                to_rename = os.path.join(self.dirname, base_dir)
                if os.path.isdir(to_rename):
                    try:
                        shutil.move(to_rename, to_delete)
                    except Exception:
                        _logger.warning('Could not move %s to %s',
                                        to_rename, to_delete)
        finally:
            compilelock.release_lock()

    def clear_unversioned(self, min_age=None):
        """
        Delete unversioned dynamic modules.

        They are deleted both from the internal dictionaries and from the
        filesystem.

        :param min_age: Minimum age to be deleted, in seconds. Defaults to
        7-day age if not provided.
        """
        if min_age is None:
            min_age = self.age_thresh_del_unversioned

        compilelock.get_lock()
        all_key_datas = self.module_hash_to_key_data.values()
        try:
            for key_data in all_key_datas:
                if not key_data.keys:
                    # May happen for broken versioned keys.
                    continue
                for key_idx, key in enumerate(key_data.keys):
                    version, rest = key
                    if version:
                        # Since the version is included in the module hash,
                        # it should not be possible to mix versioned and
                        # unversioned keys in the same KeyData object.
                        assert key_idx == 0
                        break
                if not version:
                    # Note that unversioned keys cannot be broken, so we can
                    # set do_manual_check to False to speed things up.
                    key_data.delete_keys_from(self.entry_from_key,
                                              do_manual_check=False)
                    entry = key_data.get_entry()
                    # Entry is guaranteed to be in this dictionary, because
                    # an unversioned entry should never have been loaded via
                    # refresh.
                    assert entry in self.module_from_name

                    del self.module_from_name[entry]
                    del self.module_hash_to_key_data[key_data.module_hash]

                    parent = os.path.dirname(entry)
                    assert parent.startswith(os.path.join(self.dirname, 'tmp'))
                    _rmtree(parent, msg='unversioned', level=logging.INFO,
                            ignore_nocleanup=True)

            # Sanity check: all unversioned keys should have been removed at
            # this point.
            for key in self.entry_from_key:
                assert key[0]

            time_now = time.time()
            for filename in os.listdir(self.dirname):
                if filename.startswith('tmp'):
                    try:
                        open(os.path.join(self.dirname, filename, 'key.pkl')
                                ).close()
                        has_key = True
                    except IOError:
                        has_key = False
                    if not has_key:
                        # Use the compiled file by default
                        path = module_name_from_dir(os.path.join(self.dirname,
                                                                 filename),
                                                    False)
                        # If it don't exist, use any file in the directory.
                        if path is None:
                            path = os.path.join(self.dirname, filename)
                            files = os.listdir(path)
                            if files:
                                path = os.path.join(path, files[0])
                            else:
                                # If the directory is empty skip it.
                                # They are deleted elsewhere.
                                continue
                        age = time_now - last_access_time(path)

                        # In normal case, the processus that created this
                        # directory will delete it. However, if this processus
                        # crashes, it will not be cleaned up.
                        # As we don't know if this directory is still used,
                        # we wait one week and suppose that the processus
                        # crashed, and we take care of the clean-up.
                        if age > min_age:
                            _rmtree(os.path.join(self.dirname, filename),
                                    msg='old unversioned', level=logging.INFO,
                                    ignore_nocleanup=True)
        finally:
            compilelock.release_lock()

    def _on_atexit(self):
        # Note: no need to call refresh() since it is called by clear_old().
        compilelock.get_lock()
        try:
            self.clear_old()
            self.clear_unversioned()
        finally:
            compilelock.release_lock()
        _logger.debug('Time spent checking keys: %s',
                      self.time_spent_in_check_key)


def _rmtree(parent, ignore_nocleanup=False, msg='', level=logging.DEBUG,
            ignore_if_missing=False):
    # On NFS filesystems, it is impossible to delete a directory with open
    # files in it.  So instead, some commands in this file will respond to a
    # failed rmtree() by touching a 'delete.me' file.  This file is a message
    # for a future process to try deleting the directory.
    if ignore_if_missing and not os.path.exists(parent):
        return
    try:
        if ignore_nocleanup or not config.nocleanup:
            log_msg = 'Deleting'
            if msg:
                log_msg += ' (%s)' % msg
            _logger.log(level, '%s: %s', log_msg, parent)
            shutil.rmtree(parent)
    except Exception, e:
        # If parent still exists, mark it for deletion by a future refresh()
        _logger.debug('In _rmtree, encountered exception: %s(%s)',
                      type(e), e)
        if os.path.exists(parent):
            try:
                _logger.info('placing "delete.me" in %s', parent)
                open(os.path.join(parent, 'delete.me'), 'w').close()
            except Exception, ee:
                _logger.warning("Failed to remove or mark cache directory %s "
                                "for removal %s", parent, ee)

_module_cache = None


def get_module_cache(dirname, init_args=None):
    """
    :param init_args: If not None, the (k, v) pairs in this dictionary will
    be forwarded to the ModuleCache constructor as keyword arguments.
    """
    global _module_cache
    if init_args is None:
        init_args = {}
    if _module_cache is None:
        _module_cache = ModuleCache(dirname, **init_args)
        atexit.register(_module_cache._on_atexit)
    elif init_args:
        _logger.warning('Ignoring init arguments for module cache because it '
                        'was created prior to this call')
    if _module_cache.dirname != dirname:
        _logger.warning("Returning module cache instance with different "
                "dirname (%s) than you requested (%s)",
                _module_cache.dirname, dirname)
    return _module_cache


def get_lib_extension():
    """Return the platform-dependent extension for compiled modules."""
    if sys.platform == 'win32':
        return 'pyd'
    else:
        return 'so'


def get_gcc_shared_library_arg():
    """Return the platform-dependent GCC argument for shared libraries."""
    if sys.platform == 'darwin':
        return '-dynamiclib'
    else:
        return '-shared'


def std_include_dirs():
    return (numpy.distutils.misc_util.get_numpy_include_dirs()
            + [distutils.sysconfig.get_python_inc()])


def std_lib_dirs_and_libs():
    python_inc = distutils.sysconfig.get_python_inc()
    if sys.platform == 'win32':
        # Obtain the library name from the Python version instead of the
        # installation directory, in case the user defined a custom
        # installation directory.
        python_version = distutils.sysconfig.get_python_version()
        libname = 'python' + python_version.replace('.', '')
        # Also add directory containing the Python library to the library
        # directories.
        python_lib_dirs = [os.path.join(os.path.dirname(python_inc), 'libs')]
        if "Canopy" in python_lib_dirs[0]:
            # Canopy stores libpython27.a and libmsccr90.a in this directory.
            # For some reason, these files are needed when compiling Python
            # modules, even when libpython27.lib and python27.dll are
            # available, and the *.a files have to be found earlier than
            # the other ones.
            libdir = os.path.join(sys.base_prefix, '..', '..', '..',
                                  'User', 'libs')
            for f, lib in [('libpython27.a', 'libpython 1.2'),
                           ('libmsvcr90.a', 'mingw 4.5.2')]:
                if not os.path.exists(os.path.join(libdir, f)):
                    print ("Your Python version is from Canopy. " +
                           "You need to install the package '" + lib +
                           "' from Canopy package manager."
                           )
            python_lib_dirs.insert(0, libdir)

        return [libname], python_lib_dirs

    # Suppress -lpython2.x on OS X since the `-undefined dynamic_lookup`
    # makes it unnecessary.
    elif sys.platform == 'darwin':
        return [], []
    else:
        # Typical include directory: /usr/include/python2.6
        libname = os.path.basename(python_inc)
        return [libname], []


def std_libs():
    return std_lib_dirs_and_libs()[0]


def std_lib_dirs():
    return std_lib_dirs_and_libs()[1]


def gcc_version():
    return gcc_version_str


def gcc_llvm():
    """ Detect if the g++ version used is the llvm one or not.

    It don't support all g++ parameters even if it support many of them.
    """
    if gcc_llvm.is_llvm is None:
        pass
        p = None
        try:
            p = call_subprocess_Popen(['g++', '--version'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            p.wait()
            output = p.stdout.read() + p.stderr.read()
        except OSError:
            # Typically means g++ cannot be found.
            # So it is not an llvm compiler.

            # Normally this should not happen as we should not try to
            # compile when g++ is not available. If this happen, it
            # will crash later so supposing it is not llvm is "safe".
            output = b('')
        del p
        gcc_llvm.is_llvm = b("llvm") in output
    return gcc_llvm.is_llvm
gcc_llvm.is_llvm = None


class GCC_compiler(object):
    # The equivalent flags of --march=native used by g++.
    march_flags = None

    @staticmethod
    def version_str():
        return "g++ " + gcc_version_str

    @staticmethod
    def compile_args():
        cxxflags = [flag for flag in config.gcc.cxxflags.split(' ') if flag]

        # Add the equivalent of -march=native flag.  We can't use
        # -march=native as when the compiledir is shared by multiple
        # computers (for example, if the home directory is on NFS), this
        # won't be optimum or cause crash depending if the file is compiled
        # on an older or more recent computer.
        # Those URL discuss how to find witch flags are used by -march=native.
        # http://en.gentoo-wiki.com/wiki/Safe_Cflags#-march.3Dnative
        # http://en.gentoo-wiki.com/wiki/Hardware_CFLAGS
        detect_march = GCC_compiler.march_flags is None
        if detect_march:
            for f in cxxflags:
                #If the user give an -march=X parameter, don't add one ourself
                if ((f.startswith("--march=") or f.startswith("-march="))):
                    _logger.warn(
                        "WARNING: your Theano flags `gcc.cxxflags` specify"
                        " an `-march=X` flags.\n"
                        "         It is better to let Theano/g++ find it"
                        " automatically, but we don't do it now")
                    detect_march = False
                    break

        if detect_march:
            GCC_compiler.march_flags = []

            def get_lines(cmd, parse=True):
                p = call_subprocess_Popen(cmd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          stdin=subprocess.PIPE,
                                          shell=True)
                # For mingw64 with GCC >= 4.7, passing os.devnull
                # as stdin (which is the default) results in the process
                # waiting forever without returning. For that reason,
                # we use a pipe, and use the empty string as input.
                (stdout, stderr) = p.communicate(input='')
                if p.returncode != 0:
                    return None

                lines = BytesIO(stdout + stderr).readlines()
                lines = decode_iter(lines)
                if parse:
                    selected_lines = []
                    for line in lines:
                        if ("COLLECT_GCC_OPTIONS=" in line or
                            "CFLAGS=" in line or
                            "CXXFLAGS=" in line or
                            "-march=native" in line):
                            continue
                        elif "-march=" in line:
                            selected_lines.append(line.strip())
                        elif "-mtune=" in line:
                            selected_lines.append(line.strip())
                        elif "-target-cpu" in line:
                            selected_lines.append(line.strip())
                    lines = list(set(selected_lines))  # to remove duplicate

                return lines

            # The '-' at the end is needed. Otherwise, g++ do not output
            # enough information.
            native_lines = get_lines("g++ -march=native -E -v -")
            if native_lines is None:
                _logger.info("Call to 'g++ -march=native' failed,"
                             "not setting -march flag")
                detect_march = False
            else:
                _logger.info("g++ -march=native selected lines: %s",
                             native_lines)

        if detect_march:
            if len(native_lines) != 1:
                if len(native_lines) == 0:
                    # That means we did not select the right lines, so
                    # we have to report all the lines instead
                    reported_lines = get_lines("g++ -march=native -E -v -",
                                               parse=False)
                else:
                    reported_lines = native_lines
                _logger.warn(
                    "OPTIMIZATION WARNING: Theano was not able to find the"
                    " g++ parameters that tune the compilation to your "
                    " specific CPU. This can slow down the execution of Theano"
                    " functions. Please submit the following lines to"
                    " Theano's mailing list so that we can fix this"
                    " problem:\n %s",
                    reported_lines)
            else:
                default_lines = get_lines("g++ -E -v -")
                _logger.info("g++ default lines: %s", default_lines)
                if len(default_lines) < 1:
                    _logger.warn(
                        "OPTIMIZATION WARNING: Theano was not able to find the"
                        " default g++ parameters. This is needed to tune"
                        " the compilation to your specific"
                        " CPU. This can slow down the execution of Theano"
                        " functions. Please submit the following lines to"
                        " Theano's mailing list so that we can fix this"
                        " problem:\n %s",
                        get_lines("g++ -E -v -", parse=False))
                else:
                    # Some options are actually given as "-option value",
                    # we want to treat them as only one token when comparing
                    # different command lines.
                    # Heuristic: tokens not starting with a dash should be
                    # joined with the previous one.
                    def join_options(init_part):
                        new_part = []
                        for i in range(len(init_part)):
                            p = init_part[i]
                            if p.startswith('-'):
                                p_list = [p]
                                while ((i + 1 < len(init_part)) and
                                       not init_part[i + 1].startswith('-')):
                                    # append that next part to p_list
                                    p_list.append(init_part[i + 1])
                                    i += 1
                                new_part.append(' '.join(p_list))
                            elif i == 0:
                                # The first argument does not usually start
                                # with "-", still add it
                                new_part.append(p)
                            # Else, skip it, as it was already included
                            # with the previous part.
                        return new_part

                    part = join_options(native_lines[0].split())

                    for line in default_lines:
                        if line.startswith(part[0]):
                            part2 = [p for p in join_options(line.split())
                                     if (not 'march' in p and
                                         not 'mtune' in p and
                                         not 'target-cpu' in p)]
                            new_flags = [p for p in part if p not in part2]
                            # Replace '-target-cpu value', which is an option
                            # of clang, with '-march=value', for g++
                            for i, p in enumerate(new_flags):
                                if 'target-cpu' in p:
                                    opt = p.split()
                                    if len(opt) == 2:
                                        opt_name, opt_val = opt
                                        new_flags[i] = '-march=%s' % opt_val

                            # Go back to split arguments, like
                            # ["-option", "value"],
                            # as this is the way g++ expects them split.
                            split_flags = []
                            for p in new_flags:
                                split_flags.extend(p.split())

                            GCC_compiler.march_flags = split_flags
                            break
                    _logger.info("g++ -march=native equivalent flags: %s",
                                 GCC_compiler.march_flags)

            #Add the detected -march=native equivalent flags
            cxxflags.extend(GCC_compiler.march_flags)

        #NumPy 1.7 Deprecate the old API. I updated most of the places
        #to use the new API, but not everywhere. When finished, enable
        #the following macro to assert that we don't bring new code
        #that use the old API.
        cxxflags.append("-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
        numpy_ver = [int(n) for n in numpy.__version__.split('.')[:2]]

        # numpy 1.7 deprecated the following macro but the new one didn't
        # existed in the past
        if bool(numpy_ver < [1, 7]):
            cxxflags.append("-D NPY_ARRAY_ENSURECOPY=NPY_ENSURECOPY")
            cxxflags.append("-D NPY_ARRAY_ALIGNED=NPY_ALIGNED")
            cxxflags.append("-D NPY_ARRAY_WRITEABLE=NPY_WRITEABLE")
            cxxflags.append("-D NPY_ARRAY_UPDATE_ALL=NPY_UPDATE_ALL")
            cxxflags.append("-D NPY_ARRAY_C_CONTIGUOUS=NPY_C_CONTIGUOUS")
            cxxflags.append("-D NPY_ARRAY_F_CONTIGUOUS=NPY_F_CONTIGUOUS")

        # Platform-specific flags.
        # We put them here, rather than in compile_str(), so they en up
        # in the key of the compiled module, avoiding potential conflicts.

        # Figure out whether the current Python executable is 32
        # or 64 bit and compile accordingly. This step is ignored for ARM
        # architectures in order to make Theano compatible with the Raspberry
        # Pi.
        if any([not 'arm' in flag for flag in GCC_compiler.march_flags]):
            n_bits = local_bitwidth()
            cxxflags.append('-m%d' % n_bits)
            _logger.debug("Compiling for %s bit architecture", n_bits)

        if sys.platform != 'win32':
            # Under Windows it looks like fPIC is useless. Compiler warning:
            # '-fPIC ignored for target (all code is position independent)'
            cxxflags.append('-fPIC')

        if sys.platform == 'win32' and local_bitwidth() == 64:
            # Under 64-bit Windows installation, sys.platform is 'win32'.
            # We need to define MS_WIN64 for the preprocessor to be able to
            # link with libpython.
            cxxflags.append('-DMS_WIN64')

        #DSE Patch 1 for supporting OSX frameworks; add -framework Python
        if sys.platform == 'darwin':
            cxxflags.extend(['-undefined', 'dynamic_lookup'])
            python_inc = distutils.sysconfig.get_python_inc()
            # link with the framework library *if specifically requested*
            # config.mac_framework_link is by default False, since on some mac
            # installs linking with -framework causes a Bus Error
            if (python_inc.count('Python.framework') > 0 and
                config.cmodule.mac_framework_link):
                cxxflags.extend(['-framework', 'Python'])
            if 'Anaconda' in sys.version:
                new_path = os.path.join(sys.prefix, "lib")
                new_path = os.path.realpath(new_path)
                v = os.getenv("DYLD_FALLBACK_LIBRARY_PATH", None)
                if v is not None:
                    # This will resolve symbolic links
                    v = os.path.realpath(v)

                # The python __import__ don't seam to take into account
                # the new env variable "DYLD_FALLBACK_LIBRARY_PATH"
                # when we set with os.environ['...'] = X or os.putenv()
                # So we tell the user and tell him what todo.
                if v is None or new_path not in v.split(":"):
                    raise Exception(
                        "The environment variable "
                        "'DYLD_FALLBACK_LIBRARY_PATH' does not contain "
                        "the '%s' path in its value. This will make "
                        "Theano unable to compile c code. Update "
                        "'DYLD_FALLBACK_LIBRARY_PATH' to contain the "
                        "said value, this will fix this error."
                        % new_path)

        return cxxflags

    @staticmethod
    def try_compile_tmp(src_code, tmp_prefix='', flags=(), try_run=False):
        """Try to compile (and run) a test program.

        This is useful in various occasions, to check if libraries
        or compilers are behaving as expected.

        If try_run is True, the src_code is assumed to be executable,
        and will be run.

        If try_run is False, returns the compilation status.
        If try_run is True, returns a (compile_status, run_status) pair.
        """
        if not theano.config.cxx:
            return False

        flags = list(flags)
        compilation_ok = True
        run_ok = False
        try:
            fd, path = tempfile.mkstemp(suffix='.c', prefix=tmp_prefix)
            exe_path = path[:-2]
            try:
                os.write(fd, src_code)
                os.close(fd)
                fd = None
                proc = call_subprocess_Popen(
                        ['g++', path, '-o', exe_path] + flags,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
                proc.wait()
                if proc.returncode != 0:
                    compilation_ok = False
                elif try_run:
                    # Try to execute the program
                    try:
                        proc = call_subprocess_Popen([exe_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                        proc.wait()
                        run_ok = (proc.returncode == 0)
                    finally:
                        os.remove(exe_path)
            finally:
                try:
                    if fd is not None:
                        os.close(fd)
                finally:
                    os.remove(path)

        except OSError, e:
            compilation_ok = False

        if not try_run:
            return compilation_ok
        else:
            return (compilation_ok, run_ok)

    @staticmethod
    def try_flags(flag_list):
        '''
        Try to compile a dummy file with these flags.

        Returns True if compilation was successful, False if there
        were errors.
        '''
        if not theano.config.cxx:
            return False

        code = """
        int main(int argc, char** argv)
        {
            return 0;
        }
        """
        return GCC_compiler.try_compile_tmp(code, tmp_prefix='try_flags_',
                flags=flag_list, try_run=False)

    @staticmethod
    def compile_str(module_name, src_code, location=None,
                    include_dirs=None, lib_dirs=None, libs=None,
                    preargs=None, py_module=True):
        """
        :param module_name: string (this has been embedded in the src_code

        :param src_code: a complete c or c++ source listing for the module

        :param location: a pre-existing filesystem directory where the
        cpp file and .so will be written

        :param include_dirs: a list of include directory names (each
        gets prefixed with -I)

        :param lib_dirs: a list of library search path directory names
        (each gets prefixed with -L)

        :param libs: a list of libraries to link with (each gets
        prefixed with -l)

        :param preargs: a list of extra compiler arguments

        :param py_module: if False, compile to a shared library, but do not
            import it as a Python module.

        :returns: dynamically-imported python module of the compiled code.
            (unless py_module is False, in that case returns None.)
        """
        #TODO: Do not do the dlimport in this function

        if not theano.config.cxx:
            raise MissingGXX("g++ not available! We can't compile c code.")

        if include_dirs is None:
            include_dirs = []
        if lib_dirs is None:
            lib_dirs = []
        if libs is None:
            libs = []
        if preargs is None:
            preargs = []
        else:
            preargs = list(preargs)

        include_dirs = include_dirs + std_include_dirs()
        libs = std_libs() + libs
        lib_dirs = std_lib_dirs() + lib_dirs

        # sometimes, the linker cannot find -lpython so we need to tell it
        # explicitly where it is located
        # this returns somepath/lib/python2.x
        python_lib = distutils.sysconfig.get_python_lib(plat_specific=1,
                                                        standard_lib=1)
        python_lib = os.path.dirname(python_lib)
        if python_lib not in lib_dirs:
            lib_dirs.append(python_lib)

        cppfilename = os.path.join(location, 'mod.cpp')
        cppfile = open(cppfilename, 'w')

        _logger.debug('Writing module C++ code to %s', cppfilename)

        cppfile.write(src_code)
        # Avoid gcc warning "no newline at end of file".
        if not src_code.endswith('\n'):
            cppfile.write('\n')
        cppfile.close()

        lib_filename = os.path.join(location, '%s.%s' %
                                    (module_name, get_lib_extension()))

        _logger.debug('Generating shared lib %s', lib_filename)
        cmd = ['g++', get_gcc_shared_library_arg(), '-g']

        if config.cmodule.remove_gxx_opt:
            cmd.extend(p for p in preargs if not p.startswith('-O'))
        else:
            cmd.extend(preargs)
        cmd.extend('-I%s' % idir for idir in include_dirs)
        cmd.extend(['-o', lib_filename])
        cmd.append(cppfilename)
        cmd.extend(['-L%s' % ldir for ldir in lib_dirs])
        cmd.extend(['-l%s' % l for l in libs])
        #print >> sys.stderr, 'COMPILING W CMD', cmd
        _logger.debug('Running cmd: %s', ' '.join(cmd))

        def print_command_line_error():
            # Print command line when a problem occurred.
            print >> sys.stderr, (
                    "Problem occurred during compilation with the "
                    "command line below:")
            print >> sys.stderr, ' '.join(cmd)

        try:
            p = call_subprocess_Popen(cmd, stderr=subprocess.PIPE)
            compile_stderr = decode(p.communicate()[1])
        except Exception:
            # An exception can occur e.g. if `g++` is not found.
            print_command_line_error()
            raise

        status = p.returncode

        if status:
            print '==============================='
            for i, l in enumerate(src_code.split('\n')):
                #gcc put its messages to stderr, so we add ours now
                print >> sys.stderr, '%05i\t%s' % (i + 1, l)
            print '==============================='
            print_command_line_error()
            # Print errors just below the command line.
            print compile_stderr
            # We replace '\n' by '. ' in the error message because when Python
            # prints the exception, having '\n' in the text makes it more
            # difficult to read.
            raise Exception('Compilation failed (return status=%s): %s' %
                            (status, compile_stderr.replace('\n', '. ')))
        elif config.cmodule.compilation_warning and compile_stderr:
            # Print errors just below the command line.
            print compile_stderr

        if py_module:
            #touch the __init__ file
            open(os.path.join(location, "__init__.py"), 'w').close()
            assert os.path.isfile(lib_filename)
            return dlimport(lib_filename)


def icc_module_compile_str(*args):
    raise NotImplementedError()
