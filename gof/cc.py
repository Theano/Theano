
from link import Linker
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

    def __init__(self, declare, behavior, cleanup, sub):
        self.declare = declare % sub
        behavior_sub = copy(sub)
        behavior_sub['fail'] = "{%(failure_var)s = %(id)s; goto __label_%(id)i;}" % sub
        self.behavior = behavior % behavior_sub
        # the dummy is because gcc throws an error when a label's right next to a closing
        # brace (maybe there's an ignore flag for that...)
        # we need the label even if cleanup is empty because the behavior block jumps there
        # on failure
        self.cleanup = ("__label_%(id)i:\n" + cleanup + "\ndouble __DUMMY_%(id)i;\n") % sub


def code_gen(blocks):

    decl = ""
    head = ""
    tail = ""
    for block in blocks:
        decl += block.declare
        head = head + ("\n{\n%s" % block.behavior)
        tail = ("%s\n}\n" % block.cleanup) + tail
    return decl + head + tail


def struct_gen(args, struct_builders, blocks, sub):

    struct_decl = ""
    struct_init_head = ""
    struct_init_tail = ""
    struct_cleanup = ""

    for block in struct_builders:
        struct_decl += block.declare
        struct_init_head = struct_init_head + ("\n{\n%s" % block.behavior)
        struct_init_tail = ("%s\n}\n" % block.cleanup) + struct_init_tail
        struct_cleanup += block.cleanup

    behavior = code_gen(blocks)

    storage_decl = "\n".join(["PyObject* %s;" % arg for arg in args])
    storage_set = "\n".join(["this->%s = %s;" % (arg, arg) for arg in args])
    storage_incref = "\n".join(["Py_XINCREF(%s);" % arg for arg in args])
    storage_decref = "\n".join(["Py_XDECREF(this->%s);" % arg for arg in args])
    args_names = ", ".join(args)
    args_decl = ", ".join(["PyObject* %s" % arg for arg in args])

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
    struct %%(name)s {
        PyObject* __ERROR;

        %(storage_decl)s
        %(struct_decl)s
        
        %%(name)s() {}
        ~%%(name)s(void) {
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


def get_nothing(r):
    return ""

def get_c_declare(r):
    pre = """
    PyObject* py_%(name)s;
    """
    return pre + r.c_declare()

def get_c_init(r):
    pre = "" """
    py_%(name)s = Py_None;
    """
    return pre + r.c_init()

def get_c_extract(r):
    pre = """
    py_%(name)s = PyList_GET_ITEM(storage_%(name)s, 0);
    Py_XINCREF(py_%(name)s);
    """
    return pre + r.c_extract()

def get_c_cleanup(r):
    post = """
    Py_XDECREF(py_%(name)s);
    """
    return r.c_cleanup() + post

def get_c_sync(r):
    return """
    if (!%%(failure_var)s) {
      %(sync)s
      PyObject* old = PyList_GET_ITEM(storage_%%(name)s, 0);
      Py_XINCREF(py_%%(name)s);
      PyList_SET_ITEM(storage_%%(name)s, 0, py_%%(name)s);
      Py_XDECREF(old);
    }
    """ % dict(sync = r.c_sync())

def apply_policy(policy, r):
    if isinstance(r, (list, tuple)):
        ret = ""
        for sub_policy in policy:
            ret += sub_policy(r)
    return policy(r)

def struct_result_codeblocks(result, policies, id, symbol_table, sub):
    
    name = "V%i" % id
    symbol_table[result] = name
    sub = copy(sub)
    sub['name'] = name
    sub['id'] = id
    struct_builder = CodeBlock(*[apply_policy(policy, result) for policy in policies[0]]+[sub]) # struct_declare, struct_behavior, struct_cleanup, sub)
    sub['id'] = id + 1
    block = CodeBlock(*[apply_policy(policy, result) for policy in policies[1]]+[sub]) # run_declare, run_behavior, run_cleanup, sub)

    return struct_builder, block


class CLinker(Linker):

    def __init__(self, env):
        self.env = env
        self.fetch_results()

    def fetch_results(self):
        env = self.env

        self.inputs = env.inputs
        self.outputs = env.outputs
        
        try: self.results = list(env.results())
        except AttributeError: self.results = self.inputs + self.outputs

        try: self.orphans = list(env.orphans().difference(self.outputs))
        except AttributeError: self.orphans = []

        try: self.temps = list(set(self.results).difference(self.inputs).difference(self.outputs).difference(self.orphans))
        except AttributeError: self.temps = []

        try: self.op_order = env.toposort()
        except AttributeError: self.op_order = [env]
        
    def code_gen(self, reuse_storage = True):

        if getattr(self, 'struct_code', False) and self.reuse_storage == reuse_storage:
            return self.struct_code

        env = self.env
        
        consts = []

        symbol = {}
        
        init_tasks = []
        tasks = []

        init_blocks = []
        blocks = []

        failure_var = "__failure"
        id = 1

        sub = dict(failure_var = failure_var)
       
        for result in set(self.results):

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
                except AbstractFunctionError:
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

            init_tasks.append((result, 'init', id))
            init_blocks.append(builder)

            tasks.append((result, 'get', id + 1))
            blocks.append(block)

            id += 2

        for op in self.op_order:

            ivnames, ovnames = op.c_var_names()
            sub = dict(failure_var = failure_var)
            for result, vname in zip(op.inputs + op.outputs, ivnames + ovnames):
                sub[vname] = symbol[result]

            # c_validate_update
            try: validate_behavior = op.c_validate_update()
            except AbstractFunctionError:
                validate_behavior = ""

            try: validate_cleanup = op.c_validate_update_cleanup()
            except AbstractFunctionError:
                validate_cleanup = ""

            sub['id'] = id
            blocks.append(CodeBlock("", validate_behavior, validate_cleanup, sub))
            tasks.append((op, 'validate_update', id))
            id += 1

            # c_code
            behavior = op.c_code() # this one must be implemented!

            try: cleanup = op.c_code_cleanup()
            except AbstractFunctionError:
                cleanup = ""
            
            sub['id'] = id
            blocks.append(CodeBlock("", behavior, cleanup, sub))
            tasks.append((op, 'code', id))
            id += 1

        args = []
        in_arg_order = []

        args += ["storage_%s" % symbol[result] for result in utils.uniq(self.inputs + self.outputs + self.orphans)]
        
        struct_code = struct_gen(args, init_blocks, blocks, dict(failure_var = failure_var))

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
        self.dupidx = [i for i, x in enumerate(all) if all.count(x) > 1 and all.index(x) != i]
        

    def find_task(self, failure_code):
        failure_code -= 1
        n = len(self.init_tasks)
        if failure_code < 2 * n:
            return [self.init_tasks, self.tasks][failure_code % 2][failure_code/2]
        else:
            return self.tasks[failure_code - n]
    
    def support_code(self):
        ret = ""
        for x in self.results + self.op_order:
            try: ret += x.c_support_code()
            except AbstractFunctionError: pass
        return ret

    def compile_args(self):
        ret = set()
        for x in self.results + self.op_order:
            try: ret.update(x.c_compile_args())
            except AbstractFunctionError: pass
        return ret

    def headers(self):
        ret = set()
        for x in self.results + self.op_order:
            try: ret.update(x.c_headers())
            except AbstractFunctionError: pass
        return ret
    
    def libraries(self):
        ret = set()
        for x in self.results + self.op_order:
            try: ret.update(x.c_libraries())
            except AbstractFunctionError: pass
        return ret

    def __compile__(self, inplace = False):
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
                exc_value.__thunk_trace__ = trace
                raise exc_type, exc_value, exc_trace
        return execute, in_results, out_results
    
    def cthunk_factory(self, error_storage, in_storage, out_storage):

        if not getattr(self, 'instantiate', False):
            self.code_gen()

            out_storage = [x for i, x in enumerate(out_storage) if (i+len(in_storage)) not in self.dupidx]
            in_storage = [x for i, x in enumerate(in_storage) if i not in self.dupidx]
            
            cthunk = object()
            module_name = self.hash
            mod = weave.ext_tools.ext_module(module_name)

            argnames = ["i%i" % i for i in xrange(len(in_storage))] \
                + ["o%i" % i for i in xrange(len(out_storage))] \
                + ["orph%i" % i for i in xrange(len(self.orphans))]

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

            instantiate.customize.add_support_code(self.support_code() + self.struct_code + static)
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
        
        ret = module.instantiate(error_storage, *(in_storage + out_storage + [orphan._data for orphan in self.orphans]))
        assert sys.getrefcount(ret) == 2 # refcount leak check
        return ret



class OpWiseCLinker(Linker):

    def __init__(self, env):
        self.env = env

    def make_thunk(self, inplace = False):
        if inplace:
            env = self.env
        else:
            env = self.env.clone(True)
        op_order = env.toposort()
        inputs, outputs = env.inputs, env.outputs
        env = None
        thunks = []
        for op in op_order:
            cl = CLinker(op)
            thunk, in_results, out_results = cl.make_thunk(True)
            thunks.append(thunk)

        def execute():
            for thunk in thunks:
                thunk()
        
        return execute, inputs, outputs









