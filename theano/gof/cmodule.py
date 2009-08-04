"""Generate and compile C modules for Python, 
"""
import os, tempfile, StringIO, sys, logging, subprocess, cPickle, atexit, time, shutil, stat
import distutils.sysconfig
import numpy.distutils #TODO: TensorType should handle this

import compilelock # we will abuse the lockfile mechanism when reading and writing the registry

_logger=logging.getLogger("theano.gof.cmodule")
_logger.setLevel(logging.WARN)

def error(*args):
    #sys.stderr.write('ERROR:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.error("ERROR: "+' '.join(str(a) for a in args))
def warning(*args):
    #sys.stderr.write('WARNING:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.warning("WARNING: "+' '.join(str(a) for a in args))
def info(*args):
    #sys.stderr.write('INFO:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.info("INFO: "+' '.join(str(a) for a in args))
def debug(*args):
    #sys.stderr.write('DEBUG:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.debug("DEBUG: "+' '.join(str(a) for a in args))

METH_VARARGS="METH_VARARGS"
METH_NOARGS="METH_NOARGS"

class ExtFunction(object):
    """A C function to put into a DynamicModule """

    name = ""
    """string - function's name"""

    code_block = ""
    """string - the entire code for the function.  Has the form ``static PyObject*
    <name>([...]){ ... }

    See Python's C API Reference for how to write c functions for python modules.
    """

    method = ""
    """str - calling method for this function (i.e. 'METH_VARARGS', 'METH_NOARGS')"""

    doc = ""
    """str - documentation string for this function"""

    def __init__(self, name, code_block, method, doc="undocumented"):
        self.name = name
        self.code_block = code_block
        self.method = method
        self.doc = doc

    def method_decl(self):
        """Returns the signature for this function that goes into the DynamicModule's method table"""
        return '\t{"%s", %s, %s, "%s"}' %(self.name, self.name, self.method, self.doc)

class DynamicModule(object):
    def __init__(self, name):
        self.name = name
        self.support_code = []
        self.functions = []
        self.includes = ["<Python.h>", "<iostream>"]
        self.includes.append('<numpy/arrayobject.h>') #TODO: this should come from TensorType
        self.init_blocks = ['import_array();'] #TODO: from TensorType

    def print_methoddef(self, stream):
        print >> stream, "static PyMethodDef MyMethods[] = {"
        for f in self.functions:
            print >> stream, f.method_decl(), ','
        print >> stream, "\t{NULL, NULL, 0, NULL}"
        print >> stream, "};"

    def print_init(self, stream):
        print >> stream, "PyMODINIT_FUNC init%s(void){" % self.name
        for b in self.init_blocks:
            print >> stream, '  ', b
        print >> stream, '  ', '(void) Py_InitModule("%s", MyMethods);' % self.name
        print >> stream, "}"


    def add_include(self, str):
        self.includes.append(str)
    def add_init_code(self, code):
        self.init_blocks.append(code)
    def add_support_code(self, code):
        if code not in self.support_code: #TODO: KLUDGE
            self.support_code.append(code)

    def add_function(self, fn):
        self.functions.append(fn)


    def code(self):
        sio = StringIO.StringIO()
        for inc in self.includes:
            print >> sio, "#include", inc

        print  >> sio, "//////////////////////"
        print  >> sio, "////  Support Code"
        print  >> sio, "//////////////////////"
        for sc in self.support_code:
            print >> sio, sc

        print  >> sio, "//////////////////////"
        print  >> sio, "////  Functions"
        print  >> sio, "//////////////////////"
        for f in self.functions:
            print >> sio, f.code_block

        print  >> sio, "//////////////////////"
        print  >> sio, "////  Module init"
        print  >> sio, "//////////////////////"
        self.print_methoddef(sio)
        self.print_init(sio)

        return sio.getvalue()

    def list_code(self, ofile=sys.stdout):
        """Print out the code with line numbers to `ofile` """
        for i, line in enumerate(self.code().split('\n')):
            print >> ofile, '%4i'%(i+1), line
        ofile.flush()

    #TODO: add_type

def dlimport(fullpath, suffix=None):
    """Dynamically load a .so, .pyd, .dll, or .py file

    :type fullpath: string
    :param fullpath: a fully-qualified path do a compiled python module
    :param suffix: a suffix to strip from the end of fullpath to get the import name
    :type suffix: string

    :returns: the dynamically loaded module (from __import__)

    """
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
    workdir = fullpath[:-len(module_name)- 1 - len(suffix)]
    #debug("WORKDIR", workdir)
    #debug("module_name", module_name)

    pathcopy = list(sys.path)
    sys.path = [workdir]
    try:
        rval = __import__(module_name, {}, {}, [module_name])
        if not rval:
            error('__import__ failed', fullpath)
    finally:
        sys.path = pathcopy

    assert fullpath.startswith(rval.__file__)
    return rval

def last_access_time(path):
    """Return the number of seconds since the epoch of the last access of a given file"""
    return os.stat(path)[stat.ST_ATIME]

def module_name_from_dir(dirname):
    """Scan the contents of a cache directory and return full path of the dynamic lib in it.
    """
    files = os.listdir(dirname)
    names = [file for file in files if file.endswith('.so') or file.endswith('.pyd')]
    if len(names) != 1:
        raise Exception('Failed to load dynamic libraries from dir', dirname)
    return os.path.join(dirname, names[0])

class ModuleCache(object):
    """Interface to the cache of dynamically compiled modules on disk

    Note that this interface does not assume exclusive use of the cache directory.
    It is built to handle the case where multiple programs are also using instances of this
    class to manage the same directory.


    The cache works on the basis of keys.  Keys are used to uniquely identify a dynamic module.
    Keys should be tuples of length 2: (version, rest)
    The ``rest`` can be anything hashable and picklable, that uniquely identifies the
    computation in the module.

    The ``version`` should be a hierarchy of tuples of integers.
    If the ``version`` is either 0 or (), then the corresponding module is unversioned, and
    will be deleted in an atexit() handler.
    If the ``version`` is neither 0 nor (), then the module will be kept in the cache between
    processes, but it may be deleted if another key comes
    along that has the same ``rest``, and a ``version`` that is considered higher than the
    first one.  
    
    :todo: Versioning functionality is planned for implementation later, it is not implemented
    yet.

    """

    dirname = ""
    """The working directory that is managed by this interface"""

    module_from_name = {}
    """maps module names to loaded module objects"""
    
    entry_from_key = {}
    """Maps keys to the filename of a .so/.pyd.
    """

    stats = []
    """A list with counters for the number of hits, loads, compiles issued by module_from_key()
    """

    force_fresh = False
    """True -> Ignore previously-compiled modules
    """

    loaded_key_pkl = set()
    """set of all key.pkl files that have been loaded.
    """

    def __init__(self, dirname, force_fresh=None, check_for_broken_eq=True):
        """
        :param check_for_broken_eq: A bad __eq__ implemenation can break this cache mechanism.
        This option turns on a not-too-expensive sanity check during the load of an old cache
        file.
        """
        self.dirname = dirname
        self.module_from_name = dict(self.module_from_name)
        self.entry_from_key = dict(self.entry_from_key)
        self.stats = [0, 0, 0]
        if force_fresh is None:
          self.force_fresh = self.force_fresh
        else:
          self.force_fresh = force_fresh
        #backport
        #self.force_fresh = self.force_fresh if force_fresh is None else force_fresh
        self.loaded_key_pkl = set()

        self.refresh()

        if check_for_broken_eq:
            for k0 in self.entry_from_key:
                for k1 in self.entry_from_key:
                    if k0 == k1 and not (k0 is k1):
                        warning(("The __eq__ and __hash__ functions are broken for some element"
                                " in the following two keys. The cache mechanism will say that"
                                " graphs like this need recompiling, when they could have been"
                                " retrieved):"))
                        warning("Key 0:", k0)
                        warning("Key 1:", k1)

    def refresh(self):
        """Update self.entry_from_key by walking the cache directory structure.

        Add entries that are not in the entry_from_key dictionary.

        Remove entries which have been removed from the filesystem.
        """
        compilelock.get_lock()
        try:
            # add entries that are not in the entry_from_key dictionary
            for root, dirs, files in os.walk(self.dirname):
                if os.path.join(root, 'key.pkl') in self.loaded_key_pkl:
                    continue
                if 'key.pkl' in files:
                    key_pkl = os.path.join(root, 'key.pkl')
                    debug('refresh adding', key_pkl)
                    try:
                        key = cPickle.load(file(key_pkl))
                    except:
                        error("ModuleCache.refresh() Failed to unpickle cache key", key_pkl)
                        info("Erasing broken file", key_pkl)
                        os.remove(key_pkl)
                        continue
                    if not key[0]: #if the version is False
                        warning("ModuleCache.refresh() Found unversioned key in cache, deleting it.", key_pkl)
                        info("Erasing broken file", key_pkl)
                        os.remove(key_pkl)
                        continue
                    if key not in self.entry_from_key:
                        entry = module_name_from_dir(root)
                        self.entry_from_key[key] = entry
                        # assert that we haven't already got this entry somehow
                        assert entry not in self.module_from_name
                        self.loaded_key_pkl.add(key_pkl)

            # remove entries that are not in the filesystem
            items_copy = list(self.entry_from_key.iteritems())
            for key, entry in items_copy:
                try:
                    # test to see that the file is [present and] readable
                    open(entry).close()
                    gone = False
                except IOError:
                    gone = True
                if gone:
                    # assert that we didn't have one of the deleted files
                    # loaded up and in use.  
                    # If so, it should not have been deleted.  This should be considered a
                    # failure of the OTHER process, that deleted it.
                    if entry in self.module_from_name:
                        error("The module %s that was loaded by this ModuleCache can no longer be read from file... this could lead to problems." % name)
                        del self.module_from_name[entry]

                    info("deleting ModuleCache entry", entry)
                    del self.entry_from_key[key]
                    if key[0]: 
                        #this is a versioned entry, so should have been on disk
                        self.loaded_key_pkl.remove(os.path.join(os.path.dirname(entry), 'key.pkl'))

        finally:
            compilelock.release_lock()

    def module_from_key(self, key, fn=None):
        rval = None
        try:
            _version, _rest = key
        except:
            raise ValueError("Invalid key. key must have form (version, rest)", key)
        if key in self.entry_from_key:
            # we have seen this key either in this process or previously
            #debug('OLD KEY HASH', hash(key), hash(key[1][0]), key[1][0])
            name = self.entry_from_key[key]

            if name not in self.module_from_name:
                #debug('loading name', name)
                self.module_from_name[name] = dlimport(name)
                self.stats[1] += 1
            else:
                self.stats[0] += 1
            rval = self.module_from_name[name]
        else:
            # we have never seen this key before
            location = tempfile.mkdtemp(dir=self.dirname)
            #debug("LOCATION*", location)
            try:
                module = fn(location=location)  # WILL FAIL FOR BAD C CODE
            except Exception, e:
                shutil.rmtree(location)
                #try:
                #except Exception, ee:
                    #error('failed to cleanup location', location, ee)
                raise
            name = module.__file__

            debug("Adding module to cache", key, name)
            assert name.startswith(location)
            assert name not in self.module_from_name
            assert key not in self.entry_from_key
            if _version: # save they key
                key_pkl = os.path.join(location, 'key.pkl')
                key_file = file(key_pkl, 'w')
                try:
                    if sys.platform == 'win32':
                        # Looks like there is a bug under Windows, where using the
                        # highest protocol will result in a pickle file that cannot
                        # be loaded afterwards.
                        cPickle.dump(key, key_file)
                    else:
                        cPickle.dump(key, key_file, cPickle.HIGHEST_PROTOCOL)
                    key_file.close()
                    key_broken = False
                except cPickle.PicklingError:
                    key_file.close()
                    os.remove(key_pkl)
                    warning("Cache leak due to unpickle-able key", key)
                    key_broken = True

                if not key_broken:
                    key_from_file = cPickle.load(file(key_pkl))
                    if key != key_from_file:
                        raise Exception("key not equal to unpickled version (Hint: verify the __eq__ and __hash__ functions for your Ops", (key, key_from_file))
                self.loaded_key_pkl.add(key_pkl)
            self.entry_from_key[key] = name
            self.module_from_name[name] = module

            self.stats[2] += 1
            rval = module
        #debug('stats', self.stats, sum(self.stats))
        return rval

    age_thresh = 60*60*24*31
    """The default age threshold for `clear_old` (in seconds)
    """
    def clear_old(self, age_thresh=None): #default to a 31-day age_threshold
        """
        Delete entries from the filesystem for cache entries that are too old.

        :param age_thresh: dynamic modules whose last access time is more than ``age_thresh``
        seconds ago will be erased.
        """
        if age_thresh is None:
          age_thresh = self.age_thresh

        #backport
        #age_thresh = self.age_thresh if age_thresh is None else age_thresh
        compilelock.get_lock()
        try:
            # update the age of modules that have been accessed by other processes
            self.refresh() 
            time_now = time.time()
            # the .items() is important here:
            # we need to get a copy of the whole list of keys and entries
            items_copy = list(self.entry_from_key.iteritems())
            for key, entry in items_copy: 
                age = time_now - last_access_time(entry)
                if age > age_thresh:
                    # TODO: we are assuming that modules that haven't been accessed in over
                    # age_thresh are not currently in use by other processes, but that could be
                    # false for long-running jobs...
                    assert entry not in self.module_from_name
                    del self.entry_from_key[key]
                    parent = os.path.dirname(entry)
                    assert parent.startswith(os.path.join(self.dirname, 'tmp'))
                    debug("Removing cache dir", parent)
                    shutil.rmtree(parent)

        finally:
            compilelock.release_lock()

    def clear(self):
        """
        Clear all the elements of the cache
        """
        return self.clear_old(-1.0)

    def clear_unversioned(self):
        """Delete unversioned dynamic modules from the internal dictionaries and from the
        filesystem.
        """
        items_copy = list(self.entry_from_key.iteritems())
        for key, entry in items_copy: 
            version, rest = key
            if not version:
                del self.entry_from_key[key]

                # entry is guaranteed to be in this dictionary,
                # because an unversioned entry should never have been loaded via refresh
                assert entry in self.module_from_name

                del self.module_from_name[entry] 

                parent = os.path.dirname(entry)
                assert parent.startswith(os.path.join(self.dirname, 'tmp'))
                debug("Removing unversioned dir", parent)
                shutil.rmtree(parent)
    def _on_atexit(self):
        self.refresh()
        self.clear_old()
        self.clear_unversioned()

_module_cache = None
def get_module_cache(dirname, force_fresh=None):
    global _module_cache
    if _module_cache is None:
        _module_cache = ModuleCache(dirname, force_fresh=force_fresh)
        atexit.register(_module_cache._on_atexit)
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

def gcc_module_compile_str(module_name, src_code, location=None, include_dirs=[], lib_dirs=[], libs=[],
        preargs=[], tmpdir=None):
    #TODO: don't to the dlimport in this function
    
    if preargs is None:
      preargs = []
    else:
      preargs = list(preargs)

    #backport
    #preargs= [] if preargs is None else list(preargs)
    preargs.append('-fPIC')
    no_opt = False

    include_dirs = [distutils.sysconfig.get_python_inc()] + \
                    numpy.distutils.misc_util.get_numpy_include_dirs()\
                    + include_dirs
    python_inc = distutils.sysconfig.get_python_inc()
    if sys.platform == 'win32':
        # Typical include directory: C:\Python26\include
        libname = os.path.basename(os.path.dirname(python_inc)).lower()
        # Also add directory containing the Python library to the library
        # directories.
        python_lib_dir = os.path.join(os.path.dirname(python_inc), 'libs')
        lib_dirs = [python_lib_dir] + lib_dirs
    else:
        # Typical include directory: /usr/include/python2.6
        libname = os.path.basename(python_inc)
    libs = [libname] + libs

    workdir = location

    cppfilename = os.path.join(workdir, 'mod.cpp')
    cppfile = file(cppfilename, 'w')

    debug('Writing module C++ code to', cppfilename)
    ofiles = []
    rval = None

    cppfile.write(src_code)
    cppfile.close()

    lib_filename = os.path.join(workdir, '%s.%s' %
            (module_name, get_lib_extension()))

    debug('Generating shared lib', lib_filename)
    cmd = ['g++', get_gcc_shared_library_arg(), '-g']
    if no_opt:
        cmd.extend(p for p in preargs if not p.startswith('-O'))
    else:
        cmd.extend(preargs)
    cmd.extend('-I%s'%idir for idir in include_dirs)
    cmd.extend(['-o',lib_filename]) 
    cmd.append(cppfilename)
    cmd.extend(['-L%s'%ldir for ldir in lib_dirs])
    cmd.extend(['-l%s'%l for l in libs])
    debug('Running cmd', ' '.join(cmd))

    p = subprocess.Popen(cmd)
    status = p.wait()

    if status:
        error('g++ return status', status)
    else:
        #touch the __init__ file
        file(os.path.join(workdir, "__init__.py"),'w').close()      

        rval = dlimport(lib_filename)
    return rval


def nvcc_module_compile_str(module_name, src_code, location=None, include_dirs=[], lib_dirs=[], libs=[],
        preargs=[], tmpdir=None):
    if preargs is None:
      preargs = []
    else:
      preargs = list(preargs)

    #backport
    #preargs= [] if preargs is None else list(preargs)
    preargs.append('-fPIC')
    no_opt = False


    raise NotImplementedError()

    #TODO: -O preargs should be passed globally, not to -Xcompiler

    #TODO: where to find these strings?  sys? distutils?
    include_dirs = ['/usr/include/python2.6'] + include_dirs
    libs = ['python2.6', 'cudart'] + libs
    lib_dirs = ['/usr/local/cuda/lib']+lib_dirs

    workdir = tempfile.mkdtemp(dir=location)

    cppfilename = os.path.join(workdir, 'mod.cpp') #.cpp to use g++
    cppfilename = os.path.join(workdir, 'mod.cu') #.cu to use nvopencc
    cppfile = file(cppfilename, 'w')

    debug('Writing module C++ code to', cppfilename)
    ofiles = []
    rval = None
    try:
        cppfile.write(src_code)
        cppfile.close()
        lib_filename = os.path.join(workdir, '%s.%s' %
                (module_name, get_lib_extension()))

        debug('Generating shared lib', lib_filename)
        cmd = ['nvcc', '-shared', '-g']
        cmd.extend(['-Xcompiler', ','.join(preargs)])
        cmd.extend('-I%s'%idir for idir in include_dirs)
        cmd.extend(['-o',lib_filename]) 
        cmd.append(cppfilename)
        cmd.extend(['-L%s'%ldir for ldir in lib_dirs])
        cmd.extend(['-l%s'%l for l in libs])
        debug('Running cmd', ' '.join(cmd))

        p = subprocess.Popen(cmd)
        status = p.wait()

        if status:
            warning('nvcc return status', status)
        else:
            #touch the __init__ file
            file(os.path.join(workdir, "__init__.py"),'w').close()      

            #load the module
            sys.path.insert(0, workdir)
            try:
                rval = __import__(module_name, {}, {}, [module_name])
                if not rval:
                    debug('__import__ failed')
            finally:
                del sys.path[0]

            assert pathcopy == sys.path

    finally:
        warning("TODO: cleanup")
        #os.remove(cppfilename)
        for ofile in ofiles:
            #os.remove(ofiles[0])
            pass
    return rval

def icc_module_compile_str(*args):
    raise NotImplementedError()

