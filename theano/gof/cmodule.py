"""Generate and compile C modules for Python
"""
import os, tempfile, StringIO, sys, logging, subprocess, cPickle, atexit

_logger=logging.getLogger("theano.gof.cmodule")

def warning(*args):
    #sys.stderr.write('WARNING:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.warning(' '.join(str(a) for a in args))
def info(*args):
    #sys.stderr.write('INFO:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.info(' '.join(str(a) for a in args))
def debug(*args):
    #sys.stderr.write('DEBUG:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.debug(' '.join(str(a) for a in args))

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
    """Dynamically load a .so, .dll, or .py file

    :type fullpath: string
    :param fullpath: a fully-qualified path do a compiled python module
    :param suffix: a suffix to strip from the end of fullpath to get the import name
    :type suffix: string

    :returns: the dynamically loaded module (from __import__)

    """
    if suffix is None:
        if fullpath.endswith('.so'):
            suffix = '.so'
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

class ModuleCache(object):
    def __init__(self, dirname, force_fresh=False):
        self.dirname = dirname
        self.module_from_name = {}
        self.name_from_key_filename = os.path.join(self.dirname, 'module_cache.pkl')
        self.name_from_key = {}      
        self.stats = [0, 0, 0]

        if not force_fresh:
            try:
                f = file(self.name_from_key_filename, 'r')
                self.name_from_key = cPickle.load(f)
                debug('ModuleCache loaded', len(self.name_from_key))
                f.close()
            except (IOError, EOFError):
                debug('cache load failed. Using fresh cache')
                pass

    def persist(self):
        f = file(self.name_from_key_filename, 'w')
        cPickle.dump(self.name_from_key, f)
        f.close()

    def module_from_key(self, key, fn=None):
        rval = None
        if key in self.name_from_key:
            # we have seen this key either in this process or previously
            #debug('OLD KEY HASH', hash(key), hash(key[1][0]), key[1][0])
            name = self.name_from_key[key]

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
            finally:
                #   >>TODO: erase location
                pass 

            debug('NEW KEY HASH', hash(key), hash(key[1][0]), key[1][0])
            for k,n in self.name_from_key.iteritems():
                if k == key:
                    debug("HASH OF RELOAD IS DIFFERENT", hash(k), hash(key))
                    print ''
                    print hash(k[0])
                    print hash(key[0])
                    print ''
                    print "OLD",
                    print hash(k[1][0])
                    print k[1][0].rehash()
                    print ""
                    print "NEW", hash(key[1][0]), key[1][0].rehash()
                    print ''
                    print hash(k[1][1])
                    print hash(key[1][1])
                    assert k != key
            name = module.__file__
            #debug("LOCATION**", location)
            #debug("NAME**", name)
            assert name.startswith(location)

            assert name not in self.module_from_name
            assert key not in self.name_from_key
            self.name_from_key[key] = name
            self.module_from_name[name] = module

            self.stats[2] += 1
            rval = module
        #debug('stats', self.stats, sum(self.stats))
        return rval


_module_cache = None
def get_module_cache(dirname):
    global _module_cache
    if _module_cache is None:
        _module_cache = ModuleCache(dirname, force_fresh=False)
        atexit.register(_module_cache.persist)
    return _module_cache

def gcc_module_compile_str(module_name, src_code, location=None, include_dirs=[], lib_dirs=[], libs=[],
        preargs=[], tmpdir=None):
    #TODO: don't to the dlimport in this function
    preargs= [] if preargs is None else list(preargs)
    preargs.append('-fPIC')
    no_opt = False

    #TODO: where to find these strings?  sys? distutils?
    include_dirs = ['/usr/include/python2.6'] + include_dirs
    libs = ['python2.6'] + libs

    workdir = location

    cppfilename = os.path.join(workdir, 'mod.cpp')
    cppfile = file(cppfilename, 'w')

    debug('Writing module C++ code to', cppfilename)
    ofiles = []
    rval = None
    try:
        cppfile.write(src_code)
        cppfile.close()
        lib_filename = os.path.join(workdir, '%s.so'% module_name)

        debug('Generating shared lib', lib_filename)
        cmd = ['g++', '-shared', '-g']
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

    finally:
        warning("TODO: cleanup")
        #os.remove(cppfilename)
        for ofile in ofiles:
            #os.remove(ofiles[0])
            pass
    return rval


def nvcc_module_compile_str(module_name, src_code, location=None, include_dirs=[], lib_dirs=[], libs=[],
        preargs=[], tmpdir=None):
    preargs= [] if preargs is None else list(preargs)
    preargs.append('-fPIC')
    no_opt = False


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
        lib_filename = os.path.join(workdir, '%s.so'% module_name)

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

