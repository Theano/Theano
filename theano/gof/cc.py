"""
Defines Linkers that deal with C implementations.
"""

# Python imports
from copy import copy
import re #for set_compiledir
import os, sys, StringIO
from itertools import izip


if sys.version_info[:2] >= (2,5):
    import hashlib
    def hash_from_code(msg):
        return hashlib.md5(msg).hexdigest()
else:
    import md5
    def hash_from_code(msg):
        return md5.new(msg).hexdigest()


def hash_from_file(file_path):
    """Return the MD5 hash of a file."""
    return hash_from_code(open(file_path, 'rb').read())


import theano
from theano.gof.python25 import all
from theano import config

# Note that we need to do this before importing cutils, since when there is
# no theano cache dir initialized yet, importing cutils may require compilation
# of cutils_ext.
from theano.configparser import AddConfigVar, StrParam
AddConfigVar('gcc.cxxflags',
        "Extra compiler flags for gcc",
        StrParam(""))

# gof imports
import graph
import link
import utils

from compilelock import get_lock, release_lock

import cmodule



import logging
_logger=logging.getLogger("theano.gof.cc")
_logger.setLevel(logging.WARN)

from theano.gof.callcache import CallCache

run_cthunk = None # Will be imported only when needed.


def get_module_cache(init_args=None):
    """
    :param init_args: If not None, the (k, v) pairs in this dictionary will
    be forwarded to the ModuleCache constructor as keyword arguments.
    """
    return cmodule.get_module_cache(config.compiledir, init_args=init_args)

_persistent_module_cache = None
def get_persistent_module_cache():
    global _persistent_module_cache
    if _persistent_module_cache is None:
        _persistent_module_cache = CallCache(os.path.join(config.compiledir, 'persistent_cache'))
    return _persistent_module_cache

class CodeBlock:
    """WRITEME
    Represents a computation unit composed of declare, behavior, and cleanup.
    @ivar declare: C code that declares variables for use by the computation
    @ivar behavior: C code that performs the computation
    @ivar cleanup: C code that cleans up things allocated or incref-ed in behavior
    """

    def __init__(self, declare, behavior, cleanup, sub):
        """
        Initialize a L{CodeBlock} with templatized declare, behavior and cleanup.
        The sub parameter will be used in the other arguments' templates. sub
        should contain a key called 'id' that maps to an identifier for this block.
        The identifier will be used to determine the failure code and a label
        to jump to. It should also contain a key called 'failure_var' that contains
        the name of the variable that contains the error code.
        """
        self.declare = declare
        self.behavior = behavior
        # the dummy is because gcc throws an error when a label's right next to a closing
        # brace (maybe there's an ignore flag for that...)
        # we need the label even if cleanup is empty because the behavior block jumps there
        # on failure
        self.cleanup = ("__label_%(id)i:\n"%sub + cleanup + "\ndouble __DUMMY_%(id)i;\n"%sub) #% sub


def failure_code(sub):
    """WRITEME"""
    return "{%(failure_var)s = %(id)s; goto __label_%(id)i;}" % sub


def code_gen(blocks):
    """WRITEME
    From a list of L{CodeBlock} instances, returns a string that executes them
    all in sequence. eg for C{(decl1, task1, cleanup1)} and C{(decl2, task2, cleanup2)}
    the returned string will be of the form::

        decl1
        decl2
        {
         task1
         {
          task2
          cleanup2
         }
         cleanup1
        }
    """

    decl = ""
    head = ""
    tail = ""
    for block in blocks:
        decl += block.declare
        head = head + ("\n{\n%s" % block.behavior)
        tail = ("%s\n}\n" % block.cleanup) + tail
    return decl + head + tail


def struct_gen(args, struct_builders, blocks, sub):
    """WRITEME
    Generates a struct conforming to the following specifications:
     * args -> all of the PyObject* type, stored in the struct
       they represent the storage and must be length 1 python lists.
     * struct_builders -> list of L{CodeBlock} instances such that
       * declarations are in the struct
       * behavior is in the constructor
       * cleanup is in the destructor
     * blocks -> list of CodeBlock instances such that
       * declarations, behavior and cleanup are in the run()
         method of the struct
     * sub -> dictionary used to template the struct.
       * failure_var -> must contain a variable name to use for
         the failure code.

    In a nutshell, this returns code for a struct that represents
    a function with state. The state's initialization and destruction
    are handled by struct_builders and the actual behavior of the
    function is handled by blocks.
    """

    struct_decl = ""
    struct_init_head = ""
    struct_init_tail = ""
    struct_cleanup = ""

    for block in struct_builders:
        # decl are declarations that go in the struct
        # init_head are in the constructor
        # init_tail and cleanup do the same thing, but the former will
        #     be executed if any step in the constructor fails and the
        #     latter only at destruction time.
        struct_decl += block.declare
        struct_init_head = struct_init_head + ("\n{\n%s" % block.behavior)
        struct_init_tail = ("%s\n}\n" % block.cleanup) + struct_init_tail
        struct_cleanup += block.cleanup

    behavior = code_gen(blocks)

    # declares the storage
    storage_decl = "\n".join(["PyObject* %s;" % arg for arg in args])
    # in the constructor, sets the storage to the arguments
    storage_set = "\n".join(["this->%s = %s;" % (arg, arg) for arg in args])
    # increments the storage's refcount in the constructor
    storage_incref = "\n".join(["Py_XINCREF(%s);" % arg for arg in args])
    # decrements the storage's refcount in the destructor
    storage_decref = "\n".join(["Py_XDECREF(this->%s);" % arg for arg in args])

    args_names = ", ".join(args)
    args_decl = ", ".join(["PyObject* %s" % arg for arg in args])

    # The following code stores the exception data in __ERROR, which is a special
    # field of the struct. __ERROR is a list of length 3 that holds the type, the
    # value and the traceback. After storing the error, we return the failure code
    # so we know which code block failed.
    do_return = """
        if (%(failure_var)s) {
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {err_type = Py_None;Py_INCREF(Py_None);}
            if (!err_msg) {err_msg = Py_None; Py_INCREF(Py_None);}
            if (!err_traceback) {err_traceback = Py_None; Py_INCREF(Py_None);}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {Py_XDECREF(old_err_type);}
            {Py_XDECREF(old_err_msg);}
            {Py_XDECREF(old_err_traceback);}
        }
        // The failure code is returned to index what code block failed.
        return %(failure_var)s;
        """ % sub

    sub = dict(sub)
    sub.update(locals())

    # TODO: add some error checking to make sure storage_<x> are 1-element lists
    # and __ERROR is a 3-elements list.
    struct_code = """
    struct %(name)s {
        PyObject* __ERROR;

        %(storage_decl)s
        %(struct_decl)s

        %(name)s() {}
        ~%(name)s(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, %(args_decl)s) {
            %(storage_incref)s
            %(storage_set)s
            int %(failure_var)s = 0;
            %(struct_init_head)s
            this->__ERROR = __ERROR;
            return 0;
            %(struct_init_tail)s
            %(storage_decref)s
            %(do_return)s
        }
        void cleanup(void) {
            %(struct_cleanup)s
            %(storage_decref)s
        }
        int run(void) {
            int %(failure_var)s = 0;
            %(behavior)s
            %(do_return)s
        }
    };
    """ % sub

    return struct_code


# The get_<x> functions complete the return value of r.get_<x>()
# with handling of the py_<name> variable.

def get_nothing(r, name, sub):
    """WRITEME"""
    return ""

def get_c_declare(r, name, sub):
    """WRITEME"""
    pre = """
    PyObject* py_%(name)s;
    """ % locals()
    return pre + r.type.c_declare(name, sub)

def get_c_init(r, name, sub):
    """WRITEME"""
    pre = "" """
    py_%(name)s = Py_None;
    {Py_XINCREF(py_%(name)s);}
    """ % locals()
    return pre + r.type.c_init(name, sub)

def get_c_extract(r, name, sub):
    """WRITEME"""
    pre = """
    py_%(name)s = PyList_GET_ITEM(storage_%(name)s, 0);
    {Py_XINCREF(py_%(name)s);}
    """ % locals()
    return pre + r.type.c_extract(name, sub)

def get_c_cleanup(r, name, sub):
    """WRITEME"""
    post = """
    {Py_XDECREF(py_%(name)s);}
    """ % locals()
    return r.type.c_cleanup(name, sub) + post

def get_c_sync(r, name, sub):
    """WRITEME"""
    return """
    if (!%(failure_var)s) {
      %(sync)s
      PyObject* old = PyList_GET_ITEM(storage_%(name)s, 0);
      {Py_XINCREF(py_%(name)s);}
      PyList_SET_ITEM(storage_%(name)s, 0, py_%(name)s);
      {Py_XDECREF(old);}
    }
    """ % dict(sync = r.type.c_sync(name, sub), name = name, **sub)

def apply_policy(policy, r, name, sub):
    """WRITEME
    @param policy: list of functions that map a L{Variable} to a string, or a single such function
    @type r: L{Variable}
    @return: C{policy[0](r) + policy[1](r) + ...}
    """
    if isinstance(policy, (list, tuple)):
        ret = ""
        for sub_policy in policy:
            ret += sub_policy(r, name, sub)
        return ret
    return policy(r, name, sub)



def struct_variable_codeblocks(variable, policies, id, symbol_table, sub):
    """WRITEME
    variable -> a Variable
    policies -> a pair of tuples ((declare_policy, behavior_policy, cleanup_policy), -- at construction
                                  (declare_policy, behavior_policy, cleanup_policy)) -- at execution
                the first list will produce an element of the 'struct_builders' argument in struct_gen
                the second list will produce an element of the 'blocks' argument in struct_gen
    id -> the id assigned to this variable's task in the computation
    symbol_table -> a dict that maps variables to variable names. It is not read
        by this function but a variable name for the variable is computed and added
        to the table.
    sub -> dictionary for use by L{CodeBlock}.
    """

    name = "V%i" % id
    symbol_table[variable] = name
    sub = dict(sub)
#    sub['name'] = name
    sub['id'] = id
    sub['fail'] = failure_code(sub)
    sub['py_ptr'] = "py_%s" % name
    sub['stor_ptr'] = "storage_%s" % name
    struct_builder = CodeBlock(*[apply_policy(policy, variable, name, sub)
                                 for policy in policies[0]]+[sub]) # struct_declare, struct_behavior, struct_cleanup, sub)
    sub['id'] = id + 1
    sub['fail'] = failure_code(sub)
    sub['py_ptr'] = "py_%s" % name
    sub['stor_ptr'] = "storage_%s" % name
    block = CodeBlock(*[apply_policy(policy, variable, name, sub)
                        for policy in policies[1]]+[sub]) # run_declare, run_behavior, run_cleanup, sub)

    return struct_builder, block

class CLinker(link.Linker):
    """WRITEME

    Creates C code for an env, compiles it and returns callables
    through make_thunk and make_function that make use of the compiled
    code.

    no_recycling can contain a list of Variables that belong to the env.
    If a Variable is in no_recycling, CLinker will clear the output storage
    associated to it during the computation (to avoid reusing it).
    """

    def __init__(self):
        self.env = None

    def accept(self, env, no_recycling = []):
        """WRITEME"""
        if self.env is not None and self.env is not env:
            return type(self)().accept(env, no_recycling)
            #raise Exception("Cannot accept from a Linker that is already tied to another Env.")
        self.env = env
        self.fetch_variables()
        self.no_recycling = no_recycling
        return self

    def fetch_variables(self):
        """WRITEME
        Fills the inputs, outputs, variables, orphans, temps and node_order fields.
        """
        env = self.env
        self.inputs = env.inputs
        self.outputs = env.outputs
        self.variables = graph.variables(self.inputs, self.outputs) # list(env.variables)
        # The orphans field is listified to ensure a consistent order.
        self.orphans = list(r for r in self.variables if isinstance(r, graph.Value) and r not in self.inputs) #list(env.orphans.difference(self.outputs))
        self.temps = list(set(self.variables).difference(self.inputs).difference(self.outputs).difference(self.orphans))
        self.consts = []
        self.node_order = env.toposort()

    def code_gen(self):
        """WRITEME
        Generates code for a struct that does the computation of the env and
        stores it in the struct_code field of the instance.

        If reuse_storage is True, outputs and temporaries will be stored in
        the struct so they can be reused each time a function returned by
        make_function is called, which means that the output of a call will
        be invalidated by the next. If reuse_storage is False, that problem
        is avoided.

        This method caches its computations.
        """

        if getattr(self, 'struct_code', False):
            return self.struct_code

        no_recycling = self.no_recycling

        env = self.env

        self.consts = []

        c_support_code_apply = []

        symbol = {}

        # (init_)tasks contains a list of pairs (Op/Variable, task_name)
        # e.g. (x, 'get') or (x+y, 'code')
        init_tasks = []
        tasks = []

        # (init_)blocks contain CodeBlock instances. There is a direct
        # correspondance with (init_)tasks.
        init_blocks = []
        blocks = []

        failure_var = "__failure"
        id = 1

        sub = dict(failure_var = failure_var)

        for variable in self.variables:

            # it might be possible to inline constant variables as C literals
##            if getattr(variable, 'constant', False):
            # policy = [[what to declare in the struct, what to do at construction, what to do at destruction],
            #           [what to declare in each run, what to do at the beginning of each run, what to do at the end of each run]]
            if variable in self.inputs:
                # we need to extract the new inputs at each run
                # they do not need to be relayed to Python, so we don't sync
#                 if isinstance(variable, Constant):
#                     raise TypeError("Inputs to CLinker cannot be Constant.", variable)
                policy = [[get_nothing, get_nothing, get_nothing],
                          [get_c_declare, get_c_extract, get_c_cleanup]]
            elif variable in self.orphans:
                if not isinstance(variable, graph.Value):
                    raise TypeError("All orphans to CLinker must be Value instances.", variable)
                if isinstance(variable, graph.Constant):
                    try:
                        symbol[variable] = "(" + variable.type.c_literal(variable.data) + ")"
                        self.consts.append(variable)
                        self.orphans.remove(variable)
                        continue
                    except (utils.MethodNotDefined, NotImplementedError):
                        pass
                # orphans are not inputs so we'll just get fetch them when we initialize the struct and assume they stay the same
                policy = [[get_c_declare, get_c_extract, get_c_cleanup],
                          [get_nothing, get_nothing, get_nothing]]
            elif variable in self.temps:
                # temps don't need to be extracted from Python, so we call c_init rather than c_extract
                # they do not need to be relayed to Python, so we don't sync
                if variable.type.c_is_simple() or variable in no_recycling:
                    policy = [[get_nothing, get_nothing, get_nothing],
                              [get_c_declare, get_c_init, get_c_cleanup]]
                else:
                    # it is useful for complex temps to reuse storage at each run, so we only clean up in the destructor
                    policy = [[get_c_declare, get_c_init, get_c_cleanup],
                              [get_nothing, get_nothing, get_nothing]]
            elif variable in self.outputs:
                # outputs don't need to be extracted from Python, so we call c_init rather than c_extract
                if variable.type.c_is_simple() or variable in no_recycling:
                    policy = [[get_nothing, get_nothing, get_nothing],
                              [get_c_declare, get_c_init, (get_c_sync, get_c_cleanup)]]
                else:
                    # it is useful for complex outputs to reuse storage at each run, so we only clean up in the destructor
                    policy = [[get_c_declare, get_c_init, get_c_cleanup],
                              [get_nothing, get_nothing, get_c_sync]]
            else:
                raise Exception("what the fuck")

            builder, block = struct_variable_codeblocks(variable, policy, id, symbol, sub)

            # each Variable generates two CodeBlocks, one to declare/initialize/destroy struct variables
            # and the other to declare/extract/cleanup each time the function is run.
            # Typically, only one of the two actually does anything (see all the possible combinations above)

            init_tasks.append((variable, 'init', id))
            init_blocks.append(builder)

            tasks.append((variable, 'get', id + 1))
            blocks.append(block)

            id += 2

        for node_num, node in enumerate(self.node_order):

            # We populate sub with a mapping from the variable names specified by the op's c_var_names
            # method to the actual variable names that we will use.
##            ivnames, ovnames = op.c_var_names()
            sub = dict(failure_var = failure_var)
##            for variable, vname in zip(op.inputs + op.outputs, ivnames + ovnames):
##                sub[vname] = symbol[variable]

            name = "node_%i" % node_num
            isyms, osyms = [symbol[r] for r in node.inputs], [symbol[r] for r in node.outputs]

            # c_validate_update is deprecated
            if hasattr(node.op, 'c_validate_update'):
                raise Exception("c_validate_update is deprecated, move contents to c_code", node.op)

            # Make the CodeBlock for c_code
            sub['id'] = id
            sub['fail'] = failure_code(sub)

            op = node.op
            # type-specific support code
            try:
                c_support_code_apply.append(op.c_support_code_apply(node, name))
            except utils.MethodNotDefined:
                pass
            else:
                # The following will be executed if the "try" block succeeds
                assert isinstance(c_support_code_apply[-1], basestring), (
                        str(node.op)+" didn't returned a string for c_support_code_apply")

            # emit c_code
            try:
                behavior = op.c_code(node, name, isyms, osyms, sub)
            except utils.MethodNotDefined:
                raise NotImplementedError("%s cannot produce C code" % op)
            assert isinstance(behavior, basestring), str(node.op)+" didn't returned a string for c_code"

            try:
                cleanup = op.c_code_cleanup(node, name, isyms, osyms, sub)
            except utils.MethodNotDefined:
                cleanup = ""

            _logger.info('compiling un-versioned Apply %s', str(node))

            blocks.append(CodeBlock("", behavior, cleanup, sub))
            tasks.append((node, 'code', id))
            id += 1

        # List of arg names for use in struct_gen. Note the call to uniq: duplicate inputs
        # must only be passed once because they are mapped to the same name.
        # Duplicates are defined by (a is b), rather than (a==b) since Constant instances can
        # compare equal to equivalent Constant instances.
        args = []
        args += ["storage_%s" % symbol[variable] for variable in utils.uniq(self.inputs + self.outputs + self.orphans)]

        struct_code = struct_gen(args, init_blocks, blocks, dict(failure_var = failure_var, name = "<<<<NAME>>>>"))

        # TODO: still needed? We do not use weave anymore.
        # The hash calculated on the code identifies it so weave can cache properly.
        # (the hash has to be used outside of the support code because weave does not consider changes in the support code)
        hash = hash_from_code(struct_code)

        struct_name = '__struct_compiled_op_%s' % hash
        #struct_code %= dict(name = struct_name)
        struct_code = re.sub("<<<<NAME>>>>", struct_name, struct_code)

        self.struct_code = struct_code
        self.struct_name = struct_name
        self.hash = hash
        self.args = args
        self.r2symbol = symbol
        self.init_blocks = init_blocks
        self.init_tasks = init_tasks
        self.blocks = blocks
        self.tasks = tasks
        all = self.inputs + self.outputs + self.orphans
        self.c_support_code_apply = c_support_code_apply

        if (self.init_tasks, self.tasks) != self.get_init_tasks():
            print >> sys.stderr, "init_tasks\n", self.init_tasks
            print >> sys.stderr, self.get_init_tasks()[0]
            print >> sys.stderr, "tasks\n", self.tasks
            print >> sys.stderr, self.get_init_tasks()[1]
            assert (self.init_tasks, self.tasks) == self.get_init_tasks()

        # List of indices that should be ignored when passing the arguments
        # (basically, everything that the previous call to uniq eliminated)
        self.dupidx = [i for i, x in enumerate(all) if all.count(x) > 1 and all.index(x) != i]
        return self.struct_code

    def support_code(self):
        """WRITEME
        Returns a list of support code strings that are needed by
        one or more Variables or Ops. The support code from Variables is
        added before the support code from Ops.

        This might contain duplicates.
        """
        ret = []
        # generic support code
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret.append(x.c_support_code())
            except utils.MethodNotDefined: pass
        return ret

    def compile_args(self):
        """WRITEME
        Returns a list of compile args that are needed by one
        or more Variables or Ops.

        This might contain duplicates.
        """
        ret = ["-O3"]
# this is the param the -ffast-math activate. I put the explicitly as FillMissing must disable some of them. Putting -ffast-math would make it disable all other parameter at the same time.
        ret += ["-fno-math-errno",
                #"-funsafe-math-optimizations",
                #"-fno-signaling-nans",
                #"-fcx-limited-range",
                #"-fno-rounding-math",
                #"-ffinite-math-only",
                "-Wno-unused-label",#the current code generate label event if they are not used. Could use gcc attribute for those label only
                "-Wno-unused-variable",#idem as the precedent
                "-Wno-write-strings",#generated by our code generator...
                ]
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_compile_args()
            except utils.MethodNotDefined: pass

        c_compiler = self.c_compiler()
        ret += c_compiler.compile_args()

        ret=list(set(ret))#to remove duplicate
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try:
                for i in x.c_no_compile_args():
                    try:
                        ret.remove(i)
                    except ValueError:
                        pass# in case the value is not there
            except utils.MethodNotDefined: pass
        return ret

    def headers(self):
        """WRITEME
        Returns a list of headers that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.
        """
        ret = []
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_headers()
            except utils.MethodNotDefined: pass
        return list(set(ret))

    def c_compiler(self):
        c_compiler = None
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if hasattr(x, 'c_compiler'):
                x_compiler = x.c_compiler()
            else:
                continue

            if c_compiler is None:
                c_compiler = x_compiler
            else:
                if x_compiler and (x_compiler != c_compiler):
                    raise Exception('Nodes have requested specific different compilers',
                            (c_compiler, x_compiler))
        if (c_compiler is None):
            return cmodule.GCC_compiler
        else: return c_compiler

    def header_dirs(self):
        """WRITEME
        Returns a list of lib directories that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.
        """
        ret = []
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_header_dirs()
            except utils.MethodNotDefined: pass
        return list(set(ret))

    def libraries(self):
        """WRITEME
        Returns a list of libraries that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.
        """
        ret = []
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_libraries()
            except utils.MethodNotDefined: pass
        return list(set(ret))

    def lib_dirs(self):
        """WRITEME
        Returns a list of lib directories that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.
        """
        ret = []
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_lib_dirs()
            except utils.MethodNotDefined: pass
        return list(set(ret))

    def __compile__(self, input_storage = None, output_storage = None, keep_lock=False):
        """WRITEME
        Compiles this linker's env.

        @type input_storage: list or None
        @param input_storage: list of lists of length 1. In order to use
            the thunk returned by __compile__, the inputs must be put in
            that storage. If None, storage will be allocated.
        @param output_storage: list of lists of length 1. The thunk returned
            by __compile__ will put the variables of the computation in these
            lists. If None, storage will be allocated.

        Returns: thunk, input_storage, output_storage, error_storage
        """
        error_storage = [None, None, None]
        if input_storage is None:
            input_storage = tuple([None] for variable in self.inputs)
        if output_storage is None:
            map = {}
            output_storage = []
            for variable in self.outputs:
                if variable not in map:
                    map[variable] = [None]
                output_storage.append(map[variable])
        input_storage = tuple(input_storage)
        output_storage = tuple(output_storage)
        thunk = self.cthunk_factory(error_storage,
                                    input_storage,
                                    output_storage,
                                    keep_lock=keep_lock)
        return thunk, \
            [link.Container(input, storage) for input, storage in izip(self.env.inputs, input_storage)], \
            [link.Container(output, storage, True) for output, storage in izip(self.env.outputs, output_storage)], \
            error_storage

    def get_init_tasks(self):
        init_tasks = []
        tasks = []
        id=1
        for v in self.variables:
            if v in self.consts:
                continue
            if v in self.orphans and isinstance(v, graph.Constant):
                try:
                    v.type.c_literal(v.data) #constant will be inlined, no need to get
                    continue
                except (utils.MethodNotDefined, NotImplementedError):
                    pass
            init_tasks.append((v, 'init', id))
            tasks.append((v, 'get', id+1))
            id += 2
        for node in self.node_order:
            tasks.append((node, 'code', id))
            id += 1
        return init_tasks, tasks

    def make_thunk(self, input_storage = None, output_storage = None, keep_lock=False):
        """WRITEME
        Compiles this linker's env and returns a function to perform the
        computations, as well as lists of storage cells for both the
        inputs and outputs.

        @type input_storage: list or None
        @param input_storage: list of lists of length 1. In order to use
            the thunk returned by __compile__, the inputs must be put in
            that storage. If None, storage will be allocated.
        @param output_storage: list of lists of length 1. The thunk returned
            by __compile__ will put the variables of the computation in these
            lists. If None, storage will be allocated.

        Returns: thunk, input_storage, output_storage

        The return values can be used as follows:
          f, istor, ostor = clinker.make_thunk()
          istor[0].data = first_input
          istor[1].data = second_input
          f()
          first_output = ostor[0].data
        """
        init_tasks, tasks = self.get_init_tasks()
        cthunk, in_storage, out_storage, error_storage = self.__compile__(input_storage, output_storage,
                                                                          keep_lock=keep_lock)
        res = _CThunk(cthunk, init_tasks, tasks, error_storage), in_storage, out_storage
        return res

    def cmodule_key(self):
        """Return a complete hashable signature of the module we compiled.

        This function must have the property that no two programs that compute different things
        yield the same key.

        The key returned by this function is of the form (version, signature)
        The signature has the following form:
        {{{
            'CLinker.cmodule_key', compilation args, libraries,
            header_dirs, config md5,
            (op0, input_signature0, output_signature0),
            (op1, input_signature1, output_signature1),
            ...
            (opK, input_signatureK, output_signatureK),
        }}}

        The signature is a tuple, some elements of which are sub-tuples.

        The outer tuple has a brief header, containing the compilation options
        passed to the compiler, the libraries to link against, an md5 hash
        of theano.config (for all config options where "in_c_key" is True).
        It is followed by elements for every node in the
        topological ordering of `self.env`.

        If the Op of any Apply in the Env does not have c_code_cache_ok()==True, then this
        function raises a KeyError exception.

        Input Signature
        ---------------

        Each input signature is a tuple with an element for each input
        to the corresponding Apply node.  Each element identifies the
        type of the node input, and the nature of that input in the
        graph.

        The nature of a typical variable is encoded by integer pairs ``((a,b),c)``:
        ``a`` is the topological position of the input's owner (-1 for graph inputs),
        ``b`` is the index of the variable in the owner's output list.
        ``c`` is a flag indicating whether the variable is in the no_recycling set.

        If a variable is also a graph output, then its position in the
        outputs list is also bundled with this tuple (after the b).


        The nature of a Constant instance is defined as its signature,
        together with two integers: the topological position of the
        first Apply using that Constant instance, and the lowest index
        into that Apply's inputs that refers to that Constant.  (These
        two integers are a surrogate for the id() of the Constant.
        The integers are important because merge-able constants have
        the same signature, but require separate containers in C
        code.)  The membership in no_recycling is also included in the
        signature.

        Output Signature
        ----------------

        The outputs of a node are entirely determined by the node's Op
        and the nature of the inputs, but the set of outputs that may
        be re-used by the computation (the elements of
        self.no_recycling) can affect the code that is generated.

        The format of each Op's output signature is simply a list of
        booleans, indicating whether each output is in the
        no_recycling set.

        """
        return self.cmodule_key_(self.env, self.no_recycling,
                          compile_args=self.compile_args(),
                          libraries=self.libraries(),
                          header_dirs=self.header_dirs(),
                          )
    @staticmethod
    def cmodule_key_(env, no_recycling, compile_args=[], libraries=[],
                     header_dirs=[], insert_config_md5=True):
        """
        Do the actual computation of cmodule_key in a static method
        to allow it to be reused in scalar.Composite.__eq__
        """
        order = list(env.toposort())
        #set of variables that have been computed by nodes we have
        # seen 'so far' in the loop below
        env_computed_set = set()
        env_inputs_dict = dict((i, (-1, pos)) for pos, i in enumerate(env.inputs))
        constant_ids = dict()
        op_pos = {} # Apply -> topological position

        # First we put the header, compile_args, library names and config md5
        # into the signature.
        sig = ['CLinker.cmodule_key'] # will be cast to tuple on return
        if compile_args is not None:
            # We must sort it as the order from a set are not guarantee.
            # In  particular, 2 sets with the same content can give different
            # order depending in the order you put data in it.
            # Sets are used to remove duplicate elements.
            args = sorted(compile_args)
            args = tuple(args)
            sig.append(args)
        if libraries is not None:
            # see comments for compile_args
            args = sorted(libraries)
            args = tuple(args)
            sig.append(args)

        if header_dirs is not None:
            args = sorted(header_dirs)
            args = tuple(args)
            sig.append(args)

        # IMPORTANT: The 'md5' prefix is used to isolate the compilation
        # parameters from the rest of the key. If you want to add more key
        # elements, they should be before this md5 hash if and only if they
        # can lead to a different compiled file with the same source code.
        if insert_config_md5:
            sig.append('md5:' + theano.configparser.get_config_md5())
        else:
            sig.append('md5: <omitted>')

        error_on_play = [False]
        def in_sig(i, topological_pos, i_idx):
            # assert that every input to every node is one of'
            # - an env input
            # - an output from a node in the Env
            # - a Constant

            # It is important that a variable (i)
            # yield a 'position' that reflects its role in code_gen()
            if isinstance(i, graph.Constant): #orphans
                if id(i) not in constant_ids:
                    isig = (i.signature(), topological_pos, i_idx)
                    # If the Theano constant provides a strong hash
                    # (no collision for transpose, 2, 1, 0, -1, -2,
                    # 2 element swapped...) we put this hash in the signature
                    # instead of the value. This makes the key file much
                    # smaller for big constant arrays. Before this, we saw key
                    # files up to 80M.
                    if hasattr(isig[0], "theano_hash"):
                        isig = (isig[0].theano_hash(), topological_pos, i_idx)
                    try:
                        hash(isig)
                    except Exception: #generic constants don't have a hashable signature
                        error_on_play[0] = True
                        return None
                    constant_ids[id(i)] = isig
                else:
                    isig = constant_ids[id(i)]
                #print 'SIGNATURE', i.signature()
                #return i.signature()
            elif i in env_inputs_dict:   #inputs
                isig = env_inputs_dict[i]
            else:
                if i.owner is None:
                    assert all( all(out is not None for out in o.outputs) for o in order)
                    assert all( input.owner is None for input in env.inputs)
                    raise Exception('what is this?', (i, type(i), i.clients, env))

                if i in env.outputs:
                    isig = (op_pos[i.owner], # outputs
                            i.owner.outputs.index(i),
                            env.outputs.index(i))
                else:
                    isig = (op_pos[i.owner], i.owner.outputs.index(i)) # temps
            return (isig, i in no_recycling)

        version = []
        for node_pos, node in enumerate(order):
            try:
                # Pure Ops do not have a c_code_cache_version_apply ...
                version.append(node.op.c_code_cache_version_apply(node))
            except AttributeError:
                pass
            for i in node.inputs:
                version.append(i.type.c_code_cache_version())
            for o in node.outputs:
                version.append(o.type.c_code_cache_version())

            #add the signature for this node
            sig.append((
                node.op,
                tuple((i.type, in_sig(i, node_pos, ipos))
                    for ipos,i in enumerate(node.inputs)),
                tuple(o in no_recycling for o in node.outputs)))

            if error_on_play[0]:
                # if one of the signatures is not hashable
                # then bypass the cache mechanism and
                # compile fresh every time
                return None

            op_pos[node] = node_pos
            env_computed_set.update(node.outputs)

        #crystalize the signature and version
        sig = tuple(sig)
        version = tuple(version)
        for v in version:
            if not v:
                # one of the ops or types here is unversioned,
                # so this env is entirely unversioned
                return ((), sig)
        return version, sig

    def compile_cmodule(self, location=None):
        """
        Compile the module and return it.
        """
        # Go through all steps of the compilation process.
        for step_result in self.compile_cmodule_by_step(location=location):
            pass
        # And return the output of the last step, which should be the module
        # itself.
        return step_result

    def compile_cmodule_by_step(self, location=None):
        """
        This method is a callback for `ModuleCache.module_from_key`.

        It is a generator (thus the 'by step'), so that:
            - it first yields the module's C code
            - it last yields the module itself
            - it may yield other intermediate outputs in-between if needed
              in the future (but this is not currently the case)
        """
        if location is None:
            location = cmodule.dlimport_workdir(config.compiledir)
        mod = self.build_dynamic_module()
        c_compiler = self.c_compiler()
        libs = self.libraries()
        preargs = self.compile_args()
        compiler_name = c_compiler.__name__
        if compiler_name == 'NVCC_compiler' and config.lib.amdlibm:
            # This lib does not work correctly with nvcc in device code.
            # and newer version of g++ as 4.5.1.
            # example of errors: "/usr/lib/gcc/x86_64-redhat-linux/4.5.1/include/mmintrin.h(49): error: identifier "__builtin_ia32_emms" is undefined"

            if '<amdlibm.h>' in mod.includes:
                mod.includes.remove('<amdlibm.h>')
            if '-DREPLACE_WITH_AMDLIBM' in preargs:
                preargs.remove('-DREPLACE_WITH_AMDLIBM')
            if 'amdlibm' in libs:
                libs.remove('amdlibm')
        src_code = mod.code()
        yield src_code
        get_lock()
        try:
            _logger.debug("LOCATION %s", str(location))
            try:
                module = c_compiler.compile_str(
                    module_name=mod.name,
                    src_code=src_code,
                    location=location,
                    include_dirs=self.header_dirs(),
                    lib_dirs=self.lib_dirs(),
                    libs=libs,
                    preargs=preargs)
            except Exception, e:
                e.args += (str(self.env),)
                raise
        finally:
            release_lock()

        yield module

    def build_dynamic_module(self):
        """Return a cmodule.DynamicModule instance full of the code for our env.
        """
        self.code_gen()
        module_name = self.hash

        mod = cmodule.DynamicModule(module_name)

        # The code of instantiate
        code = self.instantiate_code(1+len(self.args)) #the 1 is for error_storage
        instantiate = cmodule.ExtFunction('instantiate', code, method=cmodule.METH_VARARGS)
                #['error_storage'] + argnames,
                #local_dict = d,
                #global_dict = {})

        # Static methods that can run and destroy the struct built by instantiate.
        static = """
        int %(struct_name)s_executor(%(struct_name)s* self) {
            return self->run();
        }

        void %(struct_name)s_destructor(void* executor, void* self) {
            //printf("doing cleanup\\n");
            //fflush(stdout);
            // ((%(struct_name)s*)self)->cleanup();
            // free(self);
            delete ((%(struct_name)s*)self);
            //printf("done cleanup\\n");
            //fflush(stdout);
        }
        """ % dict(struct_name = self.struct_name)

        # We add all the support code, compile args, headers and libs we need.
        for support_code in self.support_code() + self.c_support_code_apply:
            mod.add_support_code(support_code)
        mod.add_support_code(self.struct_code)
        mod.add_support_code(static)
        mod.add_function(instantiate)
        for header in self.headers():
            mod.add_include(header)

        return mod


    def cthunk_factory(self, error_storage, in_storage, out_storage, keep_lock=False):
        """WRITEME
        error_storage -> list of length 3
        in_storage -> list of lists of length 1, one per input
        out_storage -> list of lists of length 1, one per output

        Returns a thunk that points to an instance of a C struct that
        can carry on the computation of this linker's env. That thunk,
        when executed, will fetch its inputs from in_storage, put its
        outputs in out_storage and if an error occurs will put the
        type, value and traceback of the exception in error_storage.
        """
        try:
            key = self.cmodule_key()
        except KeyError:
            key = None
        if key is None:
            # If we can't get a key, then forget the cache mechanism.
            module = self.compile_cmodule()
        else:
            module = get_module_cache().module_from_key(key=key, fn=self.compile_cmodule_by_step, keep_lock=keep_lock)

        vars = self.inputs + self.outputs + self.orphans
        # List of indices that should be ignored when passing the arguments
        # (basically, everything that the previous call to uniq eliminated)
        dupidx = [i for i, x in enumerate(vars) if vars.count(x) > 1 and vars.index(x) != i]

        out_storage = [x for i, x in enumerate(out_storage) if (i+len(in_storage)) not in dupidx]
        in_storage = [x for i, x in enumerate(in_storage) if i not in dupidx]
        orphd = [[orphan.data] for orphan in self.orphans]

        ret = module.instantiate(error_storage, *(in_storage + out_storage + orphd))

        return ret

    def instantiate_code(self, n_args):
        code = StringIO.StringIO()
        struct_name = self.struct_name
        print >> code, "static PyObject * instantiate(PyObject * self, PyObject *argtuple) {"
        print >> code, '  assert(PyTuple_Check(argtuple));'
        print >> code, '  if (%(n_args)i != PyTuple_Size(argtuple)){ ' %locals()
        print >> code, '     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected %(n_args)i, got %%i", (int)PyTuple_Size(argtuple));' %locals()
        print >> code, '     return NULL;'
        print >> code, '  }'
        print >> code, '  %(struct_name)s* struct_ptr = new %(struct_name)s();' %locals()
        print >> code, '  struct_ptr->init(', ','.join('PyTuple_GET_ITEM(argtuple, %i)'%n for n in xrange(n_args)), ');'
        print >> code, '  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&%(struct_name)s_executor), struct_ptr, %(struct_name)s_destructor);' %locals()
        print >> code, "  return thunk; }"
        return code.getvalue()

class _CThunk(object):
    """
    A thunk with a C implementation
    """

    def __init__(self, cthunk, init_tasks, tasks, error_storage):
        """
        Parameters
        ----------
        cthunk: the CObject pointer used by run_cthunk
        init_tasks: WRITEME
        tasks: WRITEME
        error_storage: WRITEME
        """
        global run_cthunk
        if run_cthunk is None:
            # Lazy import to avoid compilation when importing theano.
            from theano.gof.cutils import run_cthunk
        self.cthunk = cthunk
        self.init_tasks = init_tasks
        self.tasks = tasks
        self.error_storage = error_storage

    def find_task(self, failure_code):
        """
        Maps a failure code to the task that is associated to it.
        """
        failure_code -= 1
        n = len(self.init_tasks)
        # note that the failure code is distributed in two lists
        if failure_code < 2 * n:
            return [self.init_tasks, self.tasks][failure_code % 2][failure_code/2]
        else:
            return self.tasks[failure_code - n]

    def __call__(self):
        failure = run_cthunk(self.cthunk)
        if failure:
            task, taskname, id = self.find_task(failure)
            try:
                trace = task.trace
            except AttributeError:
                trace = ()
            try:
                exc_type, _exc_value, exc_trace = self.error_storage
                if hasattr(task, "outputs"):
                    exc_value = exc_type(_exc_value, task, task.outputs)
                else:
                    exc_value = exc_type(_exc_value, task)
                exc_value.__thunk_trace__ = trace # this can be used to retrieve the location the Op was declared
            except Exception:
                print >> sys.stderr, 'ERROR retrieving error_storage', self.error_storage
                raise



            raise exc_type, exc_value, exc_trace





class OpWiseCLinker(link.LocalLinker):
    """WRITEME
    Uses CLinker on the individual Ops that comprise an env and loops
    over them in Python. The variable is slower than a compiled version of
    the whole env, but saves on compilation time because small changes
    in the computation graph won't necessarily trigger any recompilation,
    only local changes in the Variables or Ops that are used.

    If fallback_on_perform is True, OpWiseCLinker will use an op's
    perform method if no C version can be generated.

    no_recycling can contain a list of Variables that belong to the env.
    If a Variable is in no_recycling, CLinker will clear the output storage
    associated to it prior to computation (to avoid reusing it).

    :note: This is in a sense the 'default' linker for Theano.  The overhead of using the
    OpWiseCLinker as compared with the CLinker is only noticeable for graphs of very small
    tensors (such as 20 elements or less)

    """

    __cache__ = {}

    def __init__(self,
            fallback_on_perform = True,
            allow_gc = True,
            nice_errors = True):
        self.env = None
        self.fallback_on_perform = fallback_on_perform
        self.nice_errors = nice_errors
        self.allow_gc = allow_gc

    def accept(self, env, no_recycling = []):
        if self.env is not None and self.env is not env:
            return type(self)(self.fallback_on_perform).accept(env, no_recycling)
            #raise Exception("Cannot accept from a Linker that is already tied to another Env.")
        self.env = env
        self.no_recycling = no_recycling
        return self

    def make_all(self, profiler=None, input_storage=None, output_storage=None):

        # The lock will be acquired when we compile the first
        # C code. We will keep the lock untill all the function
        # compilation will be finished. This allow to don't
        # require the lock when all c code are already compiled!
        orig_n_lock = getattr(get_lock, "n_lock", 0)
        try:

            env = self.env
            order = env.toposort()
            no_recycling = self.no_recycling

            input_storage, output_storage, storage_map = link.map_storage(
                                    env, order, input_storage, output_storage)
            if self.allow_gc:
                computed, last_user = link.gc_helper(order)
                post_thunk_old_storage = []
            else:
                post_thunk_old_storage = None

            compute_map = {}
            for k in storage_map:
                compute_map[k] = [k.owner is None]

            thunks = []
            for node in order:
                # Maker sure we use the C version of the code whenever
                # possible
                # There are ops that don't have _op_use_c_code property
                # for example ifelse (or any ops that come with their own
                # make_thunk
                old_value = getattr(node.op, '_op_use_c_code', False)
                try:
                    node.op._op_use_c_code = True
                    thunks += [node.op.make_thunk(node,
                                        storage_map,
                                        compute_map,
                                        no_recycling)]
                finally:
                    node.op._op_use_c_code = old_value

            for node_idx, node in enumerate(order):

                if self.allow_gc:
                    post_thunk_old_storage.append([storage_map[input]
                        for input in node.inputs
                        if ((input in computed) and
                            (input not in env.outputs) and
                            node == last_user[input])])

            if no_recycling is True:
                no_recycling = storage_map.values()
                no_recycling = utils.difference(no_recycling, input_storage)
            else:
                no_recycling = [storage_map[r]
                                for r in no_recycling if r not in env.inputs]

            f = link.streamline(env, thunks, order,
                    post_thunk_old_storage,
                    no_recycling=no_recycling,
                    nice_errors=self.nice_errors)

            f.allow_gc = self.allow_gc

        finally:
            # Release lock on compilation directory.
            if getattr(get_lock, "n_lock", 0) > orig_n_lock:
                release_lock()
                assert get_lock.n_lock == orig_n_lock

        return (f,
                [link.Container(input, storage)
                 for input, storage in izip(env.inputs, input_storage)],
                [link.Container(output, storage, True)
                 for output, storage in izip(env.outputs, output_storage)],
                thunks,
                order)


def _default_checker(x, y):
    """WRITEME
    Default checker for DualLinker. This checks that the
    variables contain the same data using ==.
    """
    if x[0] != y[0]:
        raise Exception("Output mismatch.",
                        {'performlinker': x[0], 'clinker': y[0]})


class DualLinker(link.Linker):
    """WRITEME
    Runs the env in parallel using PerformLinker and CLinker.

    The thunk/function produced by DualLinker uses PerformLinker as the
    "main" implementation: the inputs and outputs are fed to/taken from
    the Ops' perform. However, DualLinker also instantiates a copy of
    the env on which it runs OpWiseCLinker. At each step, the variables
    of perform and of the C implementation are verified using a checker
    function.
    """

    def __init__(self, checker=_default_checker):
        """
        Initialize a DualLinker.

        The checker argument must be a function that takes two lists
        of length 1. The first one passed will contain the output
        computed by PerformLinker and the second one the output
        computed by OpWiseCLinker. The checker should compare the data
        fields of the two variables to see if they match. By default,
        DualLinker uses ==. A custom checker can be provided to
        compare up to a certain error tolerance.

        If a mismatch occurs, the checker should raise an exception to
        halt the computation. If it does not, the computation will
        carry on and errors will snowball. The checker can sidestep
        the problem by fiddling with the data, but it should be
        careful not to share data between the two outputs (or inplace
        operations that use them will interfere).

        no_recycling can contain a list of Variables that belong to the env.
        If a Variable is in no_recycling, CLinker will clear the output storage
        associated to it during the computation (to avoid reusing it).
        """
        self.env = None
        self.checker = checker

    def accept(self, env, no_recycling=[]):
        if self.env is not None and self.env is not env:
            return type(self)(self.checker).accept(env, no_recycling)
            # raise Exception("Cannot accept from a Linker that is already "
            #                 "tied to another Env.")
        self.env = env
        self.no_recycling = no_recycling
        return self

    def make_thunk(self, **kwargs):

        env = self.env
        no_recycling = self.no_recycling

        _f, i1, o1, thunks1, order1 = link.PerformLinker().accept(env,
                                no_recycling=no_recycling).make_all(**kwargs)
        kwargs.pop('input_storage', None)
        _f, i2, o2, thunks2, order2 = OpWiseCLinker().accept(env,
                                no_recycling=no_recycling).make_all(**kwargs)

        def f():
            for input1, input2 in izip(i1, i2):
                # Set the inputs to be the same in both branches.
                # The copy is necessary in order for inplace ops not to
                # interfere.
                input2.storage[0] = copy(input1.storage[0])
            for thunk1, thunk2, node1, node2 in izip(thunks1, thunks2,
                                                     order1, order2):
                for output, storage in izip(node1.outputs, thunk1.outputs):
                    if output in no_recycling:
                        storage[0] = None
                for output, storage in izip(node2.outputs, thunk2.outputs):
                    if output in no_recycling:
                        storage[0] = None
                try:
                    thunk1()
                    thunk2()
                    for output1, output2 in izip(thunk1.outputs,
                                                 thunk2.outputs):
                        self.checker(output1, output2)
                except Exception:
                    link.raise_with_op(node1)

        return f, i1, o1
