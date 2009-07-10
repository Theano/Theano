"""
Defines Linkers that deal with C implementations.
"""

# Python imports
from copy import copy
import md5
import re #for set_compiledir
import os, sys, platform

# weave import
from scipy import weave

# gof imports
import cutils
from env import Env
import graph
import link
import utils

from compiledir import *
from compilelock import get_lock, release_lock

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

        consts = []

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
                        consts.append(variable)
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

        for node in self.node_order:

            # We populate sub with a mapping from the variable names specified by the op's c_var_names
            # method to the actual variable names that we will use.
##            ivnames, ovnames = op.c_var_names()
            sub = dict(failure_var = failure_var)
##            for variable, vname in zip(op.inputs + op.outputs, ivnames + ovnames):
##                sub[vname] = symbol[variable]

            name = "<invalid_c_thing>"
            isyms, osyms = [symbol[r] for r in node.inputs], [symbol[r] for r in node.outputs]

            # c_validate_update is deprecated
            if hasattr(node.op, 'c_validate_update'):
                raise Exception("c_validate_update is deprecated, move contents to c_code", node.op)

            # Make the CodeBlock for c_code
            sub['id'] = id
            sub['fail'] = failure_code(sub)

            op = node.op
            try: behavior = op.c_code(node, name, isyms, osyms, sub)
            except utils.MethodNotDefined:
                raise NotImplementedError("%s cannot produce C code" % op)

            try: cleanup = op.c_code_cleanup(node, name, isyms, osyms, sub)
            except utils.MethodNotDefined:
                cleanup = ""

            blocks.append(CodeBlock("", behavior, cleanup, sub))
            tasks.append((node, 'code', id))
            id += 1

        # List of arg names for use in struct_gen. Note the call to uniq: duplicate inputs
        # must only be passed once because they are mapped to the same name.
        args = []
        args += ["storage_%s" % symbol[variable] for variable in utils.uniq(self.inputs + self.outputs + self.orphans)]

        struct_code = struct_gen(args, init_blocks, blocks, dict(failure_var = failure_var, name = "<<<<NAME>>>>"))

        # The hash calculated on the code identifies it so weave can cache properly.
        # (the hash has to be used outside of the support code because weave does not consider changes in the support code)
        hash = md5.md5(struct_code).hexdigest()
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
        ret += ["-fno-math-errno", "-funsafe-math-optimizations",
                "-fno-signaling-nans", "-fcx-limited-range",
                "-fno-rounding-math", "-ffinite-math-only",
                "-Wno-unused-label",#the current code generate label event if they are not used. Could use gcc attribute for those label only
                "-Wno-unused-variable",#idem as the precedent
                "-Wno-write-strings",#generated by our code generator...
                ]
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_compile_args()
            except utils.MethodNotDefined: pass
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
        or more Variables or Ops.

        This might contain duplicates.
        """
        ret = []
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_headers()
            except utils.MethodNotDefined: pass
        return ret

    def libraries(self):
        """WRITEME
        Returns a list of libraries that are needed by one
        or more Variables or Ops.

        This might contain duplicates.
        """
        ret = []
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            try: ret += x.c_libraries()
            except utils.MethodNotDefined: pass
        return ret

    def __compile__(self, input_storage = None, output_storage = None):
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
                                    output_storage)
        return thunk, \
            [link.Container(input, storage) for input, storage in zip(self.env.inputs, input_storage)], \
            [link.Container(output, storage, True) for output, storage in zip(self.env.outputs, output_storage)], \
            error_storage

    def make_thunk(self, input_storage = None, output_storage = None):
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
        # Note: acquiring the lock here may not be necessary. However, it is
        # cheap enough that it should not matter.
        get_lock()
        try:
            cthunk, in_storage, out_storage, error_storage = self.__compile__(input_storage, output_storage)
            res = _execute(cthunk, self.init_tasks, self.tasks, error_storage), in_storage, out_storage
        except:
            release_lock()
            raise
        release_lock()
        return res

    def cthunk_factory(self, error_storage, in_storage, out_storage):
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

        get_lock()

        try:

            # check if we already compiled this
            if not getattr(self, 'instantiate', False):
    
    
                self.code_gen()
                module_name = self.hash
    
                # Eliminate duplicate inputs and outputs from the storage that we will pass to instantiate
                out_storage = [x for i, x in enumerate(out_storage) if (i+len(in_storage)) not in self.dupidx]
                in_storage = [x for i, x in enumerate(in_storage) if i not in self.dupidx]
    
                cthunk = object() # dummy so weave can get the type
                mod = weave.ext_tools.ext_module(module_name)
    
                argnames = ["i%i" % i for i in xrange(len(in_storage))] \
                    + ["o%i" % i for i in xrange(len(out_storage))] \
                    + ["orph%i" % i for i in xrange(len(self.orphans))]
    
                # The code of instantiate
                code = """
                %(struct_name)s* struct_ptr = new %(struct_name)s();
                struct_ptr->init(error_storage, %(args)s);
                PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&%(struct_name)s_executor), struct_ptr, %(struct_name)s_destructor);
                return thunk;
                // return_val = thunk; // oh my god weave why does this leak >:\
                """ % dict(struct_name = self.struct_name,
                        args = ", ".join(argnames))
    
                d = dict(error_storage = object())
                for argname in argnames:
                    d[argname] = object()
    
                instantiate = weave.ext_tools.ext_function('instantiate',
                                                        code,
                                                        ['error_storage'] + argnames,
                                                        local_dict = d,
                                                        global_dict = {})
    
                # Static methods that can run and destroy the struct built by instantiate.
                static = """
                int %(struct_name)s_executor(%(struct_name)s* self) {
                    return self->run();
                }
    
                void %(struct_name)s_destructor(void* executor, void* self) {
                    //printf("doing cleanup\\n");
                    //fflush(stdout);
                    ((%(struct_name)s*)self)->cleanup();
                    free(self);
                    //printf("done cleanup\\n");
                    //fflush(stdout);
                }
                """ % dict(struct_name = self.struct_name)
    
                # We add all the support code, compile args, headers and libs we need.
                for support_code in self.support_code():
                    instantiate.customize.add_support_code(support_code)
                instantiate.customize.add_support_code(self.struct_code)
                instantiate.customize.add_support_code(static)

                for arg in self.compile_args():
                    instantiate.customize.add_extra_compile_arg(arg)
                for header in self.headers():
                    instantiate.customize.add_header(header)
                for lib in self.libraries():
                    instantiate.customize.add_library(lib)
    
                mod.add_function(instantiate)
    
                mod.compile(location = get_compiledir())
    
                module = __import__("%s" % (module_name), {}, {}, [module_name])
    
                self.instantiate = module.instantiate
                
    
            else:
                # Eliminate duplicate inputs and outputs from the storage that we will pass to instantiate
                out_storage = [x for i, x in enumerate(out_storage) if (i+len(in_storage)) not in self.dupidx]
                in_storage = [x for i, x in enumerate(in_storage) if i not in self.dupidx]
                module_name = self.hash
                module = __import__("%s" % (module_name), {}, {}, [module_name])
    
            orphd = [[orphan.data] for orphan in self.orphans]
            ret = module.instantiate(error_storage, *(in_storage + out_storage + orphd))
            #win pdb add 3 ref count, so we disable it by default.
            #assert sys.getrefcount(ret) == 2 # refcount leak check

        except:
            release_lock()
            raise
        release_lock()

        return ret


def _execute(cthunk, init_tasks, tasks, error_storage):
    """WRITEME"""
    def find_task(failure_code):
        """
        Maps a failure code to the task that is associated to it.
        """
        failure_code -= 1
        n = len(init_tasks)
        # note that the failure code is distributed in two lists
        if failure_code < 2 * n:
            return [init_tasks, tasks][failure_code % 2][failure_code/2]
        else:
            return tasks[failure_code - n]
    def execute():
        failure = cutils.run_cthunk(cthunk)
        if failure:
            task, taskname, id = find_task(failure)
            try:
                trace = task.trace
            except AttributeError:
                trace = ()
            try:
                exc_type, _exc_value, exc_trace = error_storage
                exc_value = exc_type(_exc_value, task)
                exc_value.__thunk_trace__ = trace # this can be used to retrieve the location the Op was declared
            except:
                print >> sys.stderr, 'ERROR retrieving error_storage', error_storage
                raise
            raise exc_type, exc_value, exc_trace
    execute.cthunk = cthunk
    return execute



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

    def make_all(self, profiler = None, input_storage = None, output_storage = None):

        # Acquire lock on compilation directory.
        get_lock()
        try:

            env = self.env
            order = env.toposort()
            no_recycling = self.no_recycling

            input_storage, output_storage, storage_map = link.map_storage(env, order, input_storage, output_storage)
            if self.allow_gc:
                computed, last_user = link.gc_helper(order)
                post_thunk_old_storage = []
            else:
                post_thunk_old_storage = None

            thunks = []
            for node in order:
                node_input_storage = [storage_map[r] for r in node.inputs]
                node_output_storage = [storage_map[r] for r in node.outputs]
                try:
                    e = Env(*graph.clone(node.inputs, node.outputs))
                    e.toposort = lambda: e.nodes

                    if any(isinstance(input, graph.Value) for input in node.inputs):
                        desc = None
                    else:
                        desc = (node.op,
                                tuple(input.type for input in node.inputs),
                                tuple(input.type for input in node.inputs),
                                tuple(output in no_recycling for output in node.outputs),
                                tuple(node.inputs.count(input) for input in node.inputs))

                    try:
                        cl = self.__cache__.get(desc)
                    except Exception, exc:
                        #print >> sys.stderr, "INFO: failed to hash %s: %s. Node will not be cached." % (node, exc)
                        cl = None
                    if cl is None:
                        cl = CLinker().accept(e, [r for r, r2 in zip(e.outputs, node.outputs) if r2 in no_recycling])
                        if desc is not None:
                            try:
                                self.__cache__[desc] = cl
                            except:
                                pass

                    thunk, node_input_filters, node_output_filters = cl.make_thunk(
                        input_storage = node_input_storage,
                        output_storage = node_output_storage)
                    thunk.inputs = node_input_storage
                    thunk.outputs = node_output_storage
                    thunks.append(thunk)
                except (NotImplementedError, utils.MethodNotDefined):
                    if self.fallback_on_perform:
                        p = node.op.perform
                        thunk = lambda p = p, i = node_input_storage, o = node_output_storage, n = node: p(n, [x[0] for x in i], o)
                        thunk.inputs = node_input_storage
                        thunk.outputs = node_output_storage
                        thunk.perform = p
                        thunks.append(thunk)
                    else:
                        raise

                if self.allow_gc:
                    post_thunk_old_storage.append([storage_map[input] 
                        for input in node.inputs
                        if (input in computed) and (input not in env.outputs) and node == last_user[input]])

            if no_recycling is True:
                no_recycling = storage_map.values()
                no_recycling = utils.difference(no_recycling, input_storage)
            else:
                no_recycling = [storage_map[r] for r in no_recycling if r not in env.inputs]

            f = link.streamline(env, thunks, order, 
                    post_thunk_old_storage,
                    no_recycling = no_recycling, 
                    nice_errors = self.nice_errors)

            f.allow_gc = self.allow_gc

        finally:
            # Release lock on compilation directory.
            release_lock()

        return f, [link.Container(input, storage) for input, storage in zip(env.inputs, input_storage)], \
            [link.Container(output, storage, True) for output, storage in zip(env.outputs, output_storage)], \
            thunks, order



def _default_checker(x, y):
    """WRITEME
    Default checker for DualLinker. This checks that the
    variables contain the same data using ==.
    """
    if x[0] != y[0]:
        raise Exception("Output mismatch.", {'performlinker': x[0], 'clinker': y[0]})

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

    def __init__(self, checker = _default_checker):
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

    def accept(self, env, no_recycling = []):
        if self.env is not None and self.env is not env:
            return type(self)(self.checker).accept(env, no_recycling)
            #raise Exception("Cannot accept from a Linker that is already tied to another Env.")
        self.env = env
        self.no_recycling = no_recycling
        return self

    def make_thunk(self, **kwargs):

        env = self.env
        no_recycling = self.no_recycling

        _f, i1, o1, thunks1, order1 = link.PerformLinker().accept(env, no_recycling = no_recycling).make_all(**kwargs)
        kwargs.pop('input_storage', None)
        _f, i2, o2, thunks2, order2 =      OpWiseCLinker().accept(env, no_recycling = no_recycling).make_all(**kwargs)

        def f():
            for input1, input2 in zip(i1, i2):
                # set the inputs to be the same in both branches
                # the copy is necessary in order for inplace ops not to interfere
                input2.storage[0] = copy(input1.storage[0])
            for thunk1, thunk2, node1, node2 in zip(thunks1, thunks2, order1, order2):
                for output, storage in zip(node1.outputs, thunk1.outputs):
                    if output in no_recycling:
                        storage[0] = None
                for output, storage in zip(node2.outputs, thunk2.outputs):
                    if output in no_recycling:
                        storage[0] = None
                try:
                    thunk1()
                    thunk2()
                    for output1, output2 in zip(thunk1.outputs, thunk2.outputs):
                        self.checker(output1, output2)
                except:
                    link.raise_with_op(node1)

        return f, i1, o1

