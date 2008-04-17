
from link import Linker, raise_with_op
from copy import copy
from utils import AbstractFunctionError
import md5
import sys
import os
import platform
from scipy import weave
import cutils
import utils
import traceback


def compile_dir():
    """Return the directory in which scipy.weave should store code objects.

    If the environment variable OMEGA_COMPILEDIR is set, its value is returned.
    If not, a directory of the form $HOME/.omega/compiledir_<platform Id>.

    As a test, this function touches the file __init__.py in the returned
    directory, and raises OSError if there's a problem.

    A directory coming from OMEGA_COMPILEDIR is not created automatically, but
    a directory in $HOME/.omega is created automatically.

    This directory is appended to the sys.path search path before being
    returned, if the touch was successful.
    """
    if os.getenv('OMEGA_COMPILEDIR'):
        cachedir = os.getenv('OMEGA_COMPILEDIR')
    else:
        # use (and possibly create) a default code cache location
        platform_id = platform.platform() + '-' + platform.processor()
        import re
        platform_id = re.sub("[\(\)\s]+", "_", platform_id)
        cachedir = os.path.join(os.getenv('HOME'), '.omega', 'compiledir_'+platform_id)
        if not os.access(cachedir, os.R_OK | os.W_OK):
            #this may raise a number of problems, I think all of which are serious.
            os.makedirs(cachedir, 7<<6)
    cachedir_init = cachedir+'/__init__.py'
    touch = os.system('touch '+cachedir_init)
    if touch:
        raise OSError('touch %s returned %i' % (cachedir_init, touch))

    if cachedir not in sys.path:
        sys.path.append(cachedir)
    return cachedir



class CodeBlock:
    """
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
        self.declare = declare #% sub
#        behavior_sub = copy(sub)
#        behavior_sub['fail'] = "{%(failure_var)s = %(id)s; goto __label_%(id)i;}" % sub
        self.behavior = behavior #% behavior_sub
        # the dummy is because gcc throws an error when a label's right next to a closing
        # brace (maybe there's an ignore flag for that...)
        # we need the label even if cleanup is empty because the behavior block jumps there
        # on failure
        self.cleanup = ("__label_%(id)i:\n"%sub + cleanup + "\ndouble __DUMMY_%(id)i;\n"%sub) #% sub


def failure_code(sub):
    return "{%(failure_var)s = %(id)s; goto __label_%(id)i;}" % sub


def code_gen(blocks):
    """
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
    """
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
            if (!err_type) err_type = Py_None;
            if (!err_msg) err_msg = Py_None;
            if (!err_traceback) err_traceback = Py_None;
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            Py_XDECREF(old_err_type);
            Py_XDECREF(old_err_msg);
            Py_XDECREF(old_err_traceback);
        }
        // The failure code is returned to index what code block failed.
        return %(failure_var)s;
        """ % sub
    
    sub = copy(sub)
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
    ""
    return ""

def get_c_declare(r, name, sub):
    pre = """
    PyObject* py_%(name)s;
    """ % locals()
    return pre + r.c_declare(name, sub)

def get_c_init(r, name, sub):
    pre = "" """
    py_%(name)s = Py_None;
    """ % locals()
    return pre + r.c_init(name, sub)

def get_c_extract(r, name, sub):
    pre = """
    py_%(name)s = PyList_GET_ITEM(storage_%(name)s, 0);
    Py_XINCREF(py_%(name)s);
    """ % locals()
    return pre + r.c_extract(name, sub)

def get_c_cleanup(r, name, sub):
    post = """
    Py_XDECREF(py_%(name)s);
    """ % locals()
    return r.c_cleanup(name, sub) + post

def get_c_sync(r, name, sub):
    return """
    if (!%(failure_var)s) {
      %(sync)s
      PyObject* old = PyList_GET_ITEM(storage_%(name)s, 0);
      Py_XINCREF(py_%(name)s);
      PyList_SET_ITEM(storage_%(name)s, 0, py_%(name)s);
      Py_XDECREF(old);
    }
    """ % dict(sync = r.c_sync(name, sub), name = name, **sub)

def apply_policy(policy, r, name, sub):
    """
    @param policy: list of functions that map a L{Result} to a string, or a single such function
    @type r: L{Result}
    @return: C{policy[0](r) + policy[1](r) + ...}
    """
    if isinstance(r, (list, tuple)):
        ret = ""
        for sub_policy in policy:
            ret += sub_policy(r, name, sub)
    return policy(r, name, sub)



def struct_result_codeblocks(result, policies, id, symbol_table, sub):
    """
    result -> a Result
    policies -> a pair of tuples ((declare_policy, behavior_policy, cleanup_policy), -- at construction
                                  (declare_policy, behavior_policy, cleanup_policy)) -- at execution
                the first list will produce an element of the 'struct_builders' argument in struct_gen
                the second list will produce an element of the 'blocks' argument in struct_gen
    id -> the id assigned to this result's task in the computation
    symbol_table -> a dict that maps results to variable names. It is not read
        by this function but a variable name for the result is computed and added
        to the table.
    sub -> dictionary for use by L{CodeBlock}.
    """
    
    name = "V%i" % id
    symbol_table[result] = name
    sub = copy(sub)
#    sub['name'] = name
    sub['id'] = id
    sub['fail'] = failure_code(sub)
    struct_builder = CodeBlock(*[apply_policy(policy, result, name, sub)
                                 for policy in policies[0]]+[sub]) # struct_declare, struct_behavior, struct_cleanup, sub)
    sub['id'] = id + 1
    sub['fail'] = failure_code(sub)
    block = CodeBlock(*[apply_policy(policy, result, name, sub)
                        for policy in policies[1]]+[sub]) # run_declare, run_behavior, run_cleanup, sub)

    return struct_builder, block


class CLinker(Linker):
    """
    Creates C code for an env or an Op instance, compiles it and returns
    callables through make_thunk and make_function that make use of the
    compiled code.

    It can take an env or an Op as input.
    """

    def __init__(self, env):
        self.env = env
        self.fetch_results()

    def fetch_results(self):
        """
        Fills the inputs, outputs, results, orphans, temps and op_order fields.
        """
        
        env = self.env

        self.inputs = env.inputs
        self.outputs = env.outputs
        
        try: self.results = list(env.results())
        except AttributeError: self.results = self.inputs + self.outputs

        # The orphans field is listified to ensure a consistent order.
        try: self.orphans = list(env.orphans().difference(self.outputs))
        except AttributeError: self.orphans = []

        try: self.temps = list(set(self.results).difference(self.inputs).difference(self.outputs).difference(self.orphans))
        except AttributeError: self.temps = []

        try: self.op_order = env.toposort()
        except AttributeError: self.op_order = [env]
        
    def code_gen(self, reuse_storage = True):
        """
        Generates code for a struct that does the computation of the env and
        stores it in the struct_code field of the instance.

        If reuse_storage is True, outputs and temporaries will be stored in
        the struct so they can be reused each time a function returned by
        make_function is called, which means that the output of a call will
        be invalidated by the next. If reuse_storage is False, that problem
        is avoided.

        This method caches its computations.
        """

        if getattr(self, 'struct_code', False) and self.reuse_storage == reuse_storage:
            return self.struct_code

        env = self.env
        
        consts = []

        symbol = {}

        # (init_)tasks contains a list of pairs (Op/Result, task_name)
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
       
        for result in set(self.results):

            # it might be possible to inline constant results as C literals
            if getattr(result, 'constant', False):
                if result in self.outputs or result in self.temps:
                    raise Exception("Temporaries and outputs should not be marked constant. Check your graph.")
                try:
                    symbol[result] = result.c_literal()
                    consts.append(result)
                    if result in self.inputs:
                        print "Warning: input %s is marked as constant and has been compiled as a literal." % result
                    elif result in self.orphans:
                        self.orphans.remove(result)
                    continue
                except (AbstractFunctionError, NotImplementedError):
                    pass
            # policy = [[what to declare in the struct, what to do at construction, what to do at destruction],
            #           [what to declare in each run, what to do at the beginning of each run, what to do at the end of each run]]
            if result in self.inputs:
                # we need to extract the new inputs at each run
                # they do not need to be relayed to Python, so we don't sync
                policy = [[get_nothing, get_nothing, get_nothing],
                          [get_c_declare, get_c_extract, get_c_cleanup]]
            elif result in self.orphans:
                # orphans are not inputs so we'll just get fetch them when we initialize the struct and assume they stay the same
                policy = [[get_c_declare, get_c_extract, get_c_cleanup],
                          [get_nothing, get_nothing, get_nothing]]
            elif result in self.temps or not reuse_storage:
                # temps don't need to be extracted from Python, so we call c_init rather than c_extract
                # they do not need to be relayed to Python, so we don't sync
                if result.c_is_simple() or not reuse_storage:
                    policy = [[get_nothing, get_nothing, get_nothing],
                              [get_c_declare, get_c_init, get_c_cleanup]]
                else:
                    # it is useful for complex temps to reuse storage at each run, so we only clean up in the destructor
                    policy = [[get_c_declare, get_c_init, get_c_cleanup],
                              [get_nothing, get_nothing, get_nothing]]
            elif result in self.outputs:
                # outputs don't need to be extracted from Python, so we call c_init rather than c_extract
                if result.c_is_simple() or not reuse_storage:
                    
                    policy = [[get_nothing, get_nothing, get_nothing],
                              [get_c_declare, get_c_init, (get_c_sync, get_c_cleanup)]]
                else:
                    # it is useful for complex outputs to reuse storage at each run, so we only clean up in the destructor
                    policy = [[get_c_declare, get_c_init, get_c_cleanup],
                              [get_nothing, get_nothing, get_c_sync]]

            builder, block = struct_result_codeblocks(result, policy, id, symbol, sub)

            # each Result generates two CodeBlocks, one to declare/initialize/destroy struct variables
            # and the other to declare/extract/cleanup each time the function is run.
            # Typically, only one of the two actually does anything (see all the possible combinations above)
            
            init_tasks.append((result, 'init', id))
            init_blocks.append(builder)

            tasks.append((result, 'get', id + 1))
            blocks.append(block)

            id += 2

        for op in self.op_order:
            
            # We populate sub with a mapping from the variable names specified by the op's c_var_names
            # method to the actual variable names that we will use.
##            ivnames, ovnames = op.c_var_names()
            sub = dict(failure_var = failure_var)
##            for result, vname in zip(op.inputs + op.outputs, ivnames + ovnames):
##                sub[vname] = symbol[result]

            isyms, osyms = [symbol[r] for r in op.inputs], [symbol[r] for r in op.outputs]

            # Make the CodeBlock for c_validate_update
            sub['id'] = id
            sub['fail'] = failure_code(sub)
            
            try: validate_behavior = op.c_validate_update(isyms, osyms, sub)
            except AbstractFunctionError:
                validate_behavior = ""

            try: validate_cleanup = op.c_validate_update_cleanup(isyms, osyms, sub)
            except AbstractFunctionError:
                validate_cleanup = ""

            blocks.append(CodeBlock("", validate_behavior, validate_cleanup, sub))
            tasks.append((op, 'validate_update', id))
            id += 1

            # Make the CodeBlock for c_code
            sub['id'] = id
            sub['fail'] = failure_code(sub)

            behavior = op.c_code(isyms, osyms, sub) # this one must be implemented!

            try: cleanup = op.c_code_cleanup(isyms, osyms, sub)
            except AbstractFunctionError:
                cleanup = ""
            
            blocks.append(CodeBlock("", behavior, cleanup, sub))
            tasks.append((op, 'code', id))
            id += 1

        # List of arg names for use in struct_gen. Note the call to uniq: duplicate inputs
        # must only be passed once because they are mapped to the same name.
        args = []
        args += ["storage_%s" % symbol[result] for result in utils.uniq(self.inputs + self.outputs + self.orphans)]
        
        struct_code = struct_gen(args, init_blocks, blocks, dict(failure_var = failure_var, name = "%(name)s"))

        # The hash calculated on the code identifies it so weave can cache properly.
        # (the hash has to be used outside of the support code because weave does not consider changes in the support code)
        hash = md5.md5(struct_code).hexdigest()
        struct_name = '__struct_compiled_op_%s' % hash
        struct_code %= dict(name = struct_name)

        self.struct_code = struct_code
        self.reuse_storage = reuse_storage
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
    
    def support_code(self):
        """
        Returns a list of support code strings that are needed by
        one or more Results or Ops. The support code from Results is
        added before the support code from Ops.

        This might contain duplicates.
        """
        ret = []
        for x in self.results + self.op_order:
            try: ret.append(x.c_support_code())
            except AbstractFunctionError: pass
        return ret

    def compile_args(self):
        """
        Returns a list of compile args that are needed by one
        or more Results or Ops.

        This might contain duplicates.
        """
        ret = []
        for x in self.results + self.op_order:
            try: ret += x.c_compile_args()
            except AbstractFunctionError: pass
        return ret

    def headers(self):
        """
        Returns a list of headers that are needed by one
        or more Results or Ops.

        This might contain duplicates.
        """
        ret = []
        for x in self.results + self.op_order:
            try: ret += x.c_headers()
            except AbstractFunctionError: pass
        return ret
    
    def libraries(self):
        """
        Returns a list of libraries that are needed by one
        or more Results or Ops.

        This might contain duplicates.
        """
        ret = []
        for x in self.results + self.op_order:
            try: ret += x.c_libraries()
            except AbstractFunctionError: pass
        return ret

    def __compile__(self, inplace = False):
        """
        Compiles this linker's env. If inplace is True, it will use the
        Results contained in the env, if it is False it will copy the
        input and output Results.

        Returns: thunk, in_results, out_results, error_storage
        """
        if inplace:
            in_results = self.inputs
            out_results = self.outputs
        else:
            in_results = [copy(input) for input in self.inputs]
            out_results = [copy(output) for output in self.outputs]
        error_storage = [None, None, None]
        thunk = self.cthunk_factory(error_storage,
                                    [result._data for result in in_results],
                                    [result._data for result in out_results])
        if not inplace:
            for r in in_results + out_results:
                r._role = None # we just need the wrapper, not the (copied) graph associated to it
        return thunk, in_results, out_results, error_storage

    def make_thunk(self, inplace = False):
        cthunk, in_results, out_results, error_storage = self.__compile__(inplace)
        def execute():
            failure = cutils.run_cthunk(cthunk)
            if failure:
                task, taskname, id = self.find_task(failure)
                try:
                    trace = task.trace
                except AttributeError:
                    trace = ()
                exc_type, _exc_value, exc_trace = error_storage
                exc_value = exc_type(_exc_value, task)
                exc_value.__thunk_trace__ = trace # this can be used to retrieve the location the Op was declared
                raise exc_type, exc_value, exc_trace
        return execute, in_results, out_results
    
    def cthunk_factory(self, error_storage, in_storage, out_storage):
        """
        error_storage -> list of length 3
        in_storage -> list of lists of length 1, one per input
        out_storage -> list of lists of length 1, one per output

        Returns a thunk that points to an instance of a C struct that
        can carry on the computation of this linker's env. That thunk,
        when executed, will fetch its inputs from in_storage, put its
        outputs in out_storage and if an error occurs will put the
        type, value and traceback of the exception in error_storage.
        """

        # check if we already compiled this
        if not getattr(self, 'instantiate', False):
            
            self.code_gen()

            # Eliminate duplicate inputs and outputs from the storage that we will pass to instantiate
            out_storage = [x for i, x in enumerate(out_storage) if (i+len(in_storage)) not in self.dupidx]
            in_storage = [x for i, x in enumerate(in_storage) if i not in self.dupidx]
            
            cthunk = object() # dummy so weave can get the type
            module_name = self.hash
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
                ((%(struct_name)s*)self)->cleanup();
                free(self);
            }
            """ % dict(struct_name = self.struct_name)

            # We add all the support code, compile args, headers and libs we need.
            for support_code in self.support_code():
                instantiate.customize.add_support_code(support_code)
            instantiate.customize.add_support_code(self.struct_code)
            instantiate.customize.add_support_code(static)
            instantiate.customize.add_extra_compile_arg("-w")
            for arg in self.compile_args():
                instantiate.customize.add_extra_compile_arg(arg)
            for header in self.headers():
                instantiate.customize.add_header(header)
            for lib in self.libraries():
                instantiate.customize.add_library(lib)

            mod.add_function(instantiate)
            mod.compile(location = compile_dir())
            module = __import__("%s" % (module_name), {}, {}, [module_name])

            self.instantiate = module.instantiate
        else:            
            # Eliminate duplicate inputs and outputs from the storage that we will pass to instantiate
            out_storage = [x for i, x in enumerate(out_storage) if (i+len(in_storage)) not in self.dupidx]
            in_storage = [x for i, x in enumerate(in_storage) if i not in self.dupidx]

        ret = module.instantiate(error_storage, *(in_storage + out_storage + [orphan._data for orphan in self.orphans]))
        assert sys.getrefcount(ret) == 2 # refcount leak check
        return ret



class OpWiseCLinker(Linker):
    """
    Uses CLinker on the individual Ops that comprise an env and loops
    over them in Python. The result is slower than a compiled version of
    the whole env, but saves on compilation time because small changes
    in the computation graph won't necessarily trigger any recompilation,
    only local changes in the Results or Ops that are used.

    If fallback_on_perform is True, OpWiseCLinker will use an op's
    perform method if no C version can be generated.
    """

    def __init__(self, env, fallback_on_perform = True):
        self.env = env
        self.fallback_on_perform = fallback_on_perform

    def make_thunk(self, inplace = False, profiler = None):
        if inplace:
            env = self.env
        else:
            env = self.env.clone(True)
        op_order = env.toposort()
        inputs, outputs = env.inputs, env.outputs
        env = None
        thunks = []
        for op in op_order:
            try:
                cl = CLinker(op)
                thunk, in_results, out_results = cl.make_thunk(True)
                thunks.append(thunk)
            except AbstractFunctionError:
                if self.fallback_on_perform:
                    thunks.append(op.perform)
                else:
                    raise
        
        if profiler is None:
            def f():
                try:
                    for thunk, op in zip(thunks, op_order):
                        thunk()
                except:
                    raise_with_op(op)
        else:
            def f():
                def g():
                    for thunk, op in zip(thunks, op_order):
                        profiler.profile_op(thunk, op)
                profiler.profile_env(g, env)
            f.profiler = profiler

        return f, inputs, outputs




def _default_checker(x, y):
    """
    Default checker for DualLinker. This checks that the
    results contain the same data using ==.
    """
    if x.data != y.data:
        raise Exception("Output mismatch.", {'performlinker': x.data, 'clinker': y.data})

class DualLinker(Linker):
    """
    Runs the env in parallel using PerformLinker and CLinker.

    The thunk/function produced by DualLinker uses PerformLinker as the
    "main" implementation: the inputs and outputs are fed to/taken from
    the Ops' perform. However, DualLinker also instantiates a copy of
    the env on which it runs OpWiseCLinker. At each step, the results
    of perform and of the C implementation are verified using a checker
    function.
    """

    def __init__(self, env, checker = _default_checker):
        """
        Initialize a DualLinker.
        
        The checker argument must be a function that takes two Result
        instances. The first one passed will be the output computed by
        PerformLinker and the second one the output computed by
        OpWiseCLinker. The checker should compare the data fields of
        the two results to see if they match. By default, DualLinker
        uses ==. A custom checker can be provided to compare up to a
        certain error tolerance.

        If a mismatch occurs, the checker should raise an exception to
        halt the computation. If it does not, the computation will
        carry on and errors will snowball. The checker can sidestep
        the problem by fiddling with the data, but it should be
        careful not to share data between the two outputs (or inplace
        operations that use them will interfere).
        """
        self.env = env
        self.checker = checker

    def make_thunk(self, inplace = False):
        if inplace:
            env1 = self.env
        else:
            env1 = self.env.clone(True)
        env2, equiv = env1.clone_get_equiv(True)

        op_order_1 = env1.toposort()
        op_order_2 = [equiv[op.outputs[0]].owner for op in op_order_1] # we need to have the exact same order so we can compare each step

        def c_make_thunk(op):
            try:
                return CLinker(op).make_thunk(True)[0]
            except AbstractFunctionError:
                return op.perform
        
        thunks1 = [op.perform for op in op_order_1]
        thunks2 = [c_make_thunk(op) for op in op_order_2]
        
        def f():
            for input1, input2 in zip(env1.inputs, env2.inputs):
                # set the inputs to be the same in both branches
                # the copy is necessary in order for inplace ops not to interfere
                input2.data = copy(input1.data)
            for thunk1, thunk2, op1, op2 in zip(thunks1, thunks2, op_order_1, op_order_2):
                try:
                    thunk1()
                    thunk2()
                    for output1, output2 in zip(op1.outputs, op2.outputs):
                        self.checker(output1, output2)
                except:
                    raise_with_op(op1)
#                     exc_type, exc_value, exc_trace = sys.exc_info()
#                     try:
#                         trace = op1.trace
#                     except AttributeError:
#                         trace = ()
#                     exc_value.__thunk_trace__ = trace
#                     exc_value.args = exc_value.args + (op1, )
#                     raise exc_type, exc_value, exc_trace

        return f, env1.inputs, env1.outputs






