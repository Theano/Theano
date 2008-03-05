
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
    # we're borrowing the references to the storage pointers because Python
    # has (needs) references to them to feed inputs or get the results
    storage_set = "\n".join(["this->%s = %s;" % (arg, arg) for arg in args])
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
            %(storage_set)s
            int %(failure_var)s = 0;
            %(struct_init_head)s
            this->__ERROR = __ERROR;
            return 0;
            %(struct_init_tail)s
            %(do_return)s
            return %(failure_var)s;
        }
        void cleanup(void) {
            %(struct_cleanup)s
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

    def __init__(self, env, inputs = None, outputs = None):
        self.env = env
        self.inputs = inputs
        self.outputs = outputs

    def fetch_results(self):
        env = self.env
        results = env.results()

        if self.inputs:
            assert set(self.inputs) == set(env.inputs)
            inputs = self.inputs
        else:
            inputs = env.inputs

        if self.outputs:
            assert set(self.outputs) == set(env.outputs)
            outputs = self.outputs
        else:
            outputs = env.outputs
            
        outputs = env.outputs
        orphans = env.orphans()
        temps = results.difference(inputs).difference(outputs).difference(orphans)
        return results, inputs, outputs, orphans, temps
        
    def code_gen(self, reuse_storage = True):

        env = self.env
        op_order = env.toposort()

        results, inputs, outputs, orphans, temps = self.fetch_results()
        
        consts = []

        symbol = {}
        
        init_tasks = []
        tasks = []

        init_blocks = []
        blocks = []

        failure_var = "__failure"
        id = 0

        sub = dict(failure_var = failure_var)

        for result in results:
            if getattr(result, 'constant', False):
                if result in outputs or result in temps:
                    raise Exception("Temporaries and outputs should not be marked constant. Check your graph.")
                try:
                    symbol[result] = result.c_literal()
                    consts.append(result)
                    if result in inputs:
                        print "Warning: input %s is marked as constant and has been compiled as a literal." % result
                    elif result in orphans:
                        orphans.remove(result)
                    continue
                except AbstractFunctionError:
                    pass
            # policy = [[what to declare in the struct, what to do at construction, what to do at destruction],
            #           [what to declare in each run, what to do at the beginning of each run, what to do at the end of each run]]
            if result in inputs:
                # we need to extract the new inputs at each run
                # they do not need to be relayed to Python, so we don't sync
                policy = [[get_nothing, get_nothing, get_nothing],
                          [get_c_declare, get_c_extract, get_c_cleanup]]
            elif result in orphans:
                # orphans are not inputs so we'll just get fetch them when we initialize the struct and assume they stay the same
                policy = [[get_c_declare, get_c_extract, get_c_cleanup],
                          [get_nothing, get_nothing, get_nothing]]
            elif result in temps or not reuse_storage:
                # temps don't need to be extracted from Python, so we call c_init rather than c_extract
                # they do not need to be relayed to Python, so we don't sync
                if result.c_is_simple() or not reuse_storage:
                    policy = [[get_nothing, get_nothing, get_nothing],
                              [get_c_declare, get_c_init, get_c_cleanup]]
                else:
                    # it is useful for complex temps to reuse storage at each run, so we only clean up in the destructor
                    policy = [[get_c_declare, get_c_init, get_c_cleanup],
                              [get_nothing, get_nothing, get_nothing]]
            elif result in outputs:
                # outputs don't need to be extracted from Python, so we call c_init rather than c_extract
                if result.c_is_simple() or not reuse_storage:
                    
                    policy = [[get_nothing, get_nothing, get_nothing],
                              [get_c_declare, get_c_init, (get_c_sync, get_c_cleanup)]]
                else:
                    # it is useful for complex outputs to reuse storage at each run, so we only clean up in the destructor
                    policy = [[get_c_declare, get_c_init, get_c_cleanup],
                              [get_nothing, get_nothing, get_c_sync]]
            
            builder, block = struct_result_codeblocks(result, policy, id, symbol, sub)

            init_tasks.append((result, 'init'))
            init_blocks.append(builder)

            tasks.append((result, 'get'))
            blocks.append(block)

            id += 2

        print symbol
        
        for op in op_order:

            ivnames, ovnames = op.c_var_names()
            sub = dict(failure_var = failure_var)
            for result, vname in zip(op.inputs + op.outputs, ivnames + ovnames):
                sub[vname] = symbol[result]

            # c_validate_update
            try: validate_behavior = op.c_validate_update()
            except AbstractFunctionError:
                validate_behavior = ""

            try: validate_behavior = op.c_validate_update_cleanup()
            except AbstractFunctionError:
                validate_cleanup = ""

            sub['id'] = id
            blocks.append(CodeBlock("", validate_behavior, validate_cleanup, sub))
            tasks.append((op, 'validate_update'))
            id += 1

            # c_code
            behavior = op.c_code() # this one must be implemented!

            try: cleanup = op.c_code_cleanup()
            except AbstractFunctionError:
                cleanup = ""
            
            sub['id'] = id
            blocks.append(CodeBlock("", behavior, cleanup, sub))
            tasks.append((op, 'code'))
            id += 1

        args = []
        in_arg_order = []
        for result in list(inputs):
            in_arg_order.append(result)
            args.append("storage_%s" % symbol[result])
        out_arg_order = []
        for result in list(outputs):
            out_arg_order.append(result)
            args.append("storage_%s" % symbol[result])
        orphan_arg_order = []
        for result in list(orphans):
            orphan_arg_order.append(result)
            args.append("storage_%s" % symbol[result])
        struct_code = struct_gen(args, init_blocks, blocks, dict(failure_var = failure_var))

        hash = md5.md5(struct_code).hexdigest()
        struct_name = '__struct_compiled_op_%s' % hash
        struct_code %= dict(name = struct_name)

        self.struct_code = struct_code
        self.struct_name = struct_name
        self.hash = hash
        self.args = args
        self.inputs = in_arg_order
        self.outputs = out_arg_order
        self.orphans = orphan_arg_order
        self.r2symbol = symbol
        self.init_blocks = init_blocks
        self.init_tasks = init_tasks
        self.blocks = blocks
        self.tasks = tasks
        
        return struct_code

    def find_task(self, failure_code):
        n = len(self.init_tasks)
        if failure_code < 2 * n:
            return [self.init_tasks, self.tasks][failure_code % 2][failure_code/2]
        else:
            return self.tasks[failure_code - n]
    
    def support_code(self):
        ret = ""
        for x in self.env.results().union(self.env.ops()):
            try: ret += x.c_support_code()
            except AbstractFunctionError: pass
        return ret

    def compile_args(self):
        ret = set()
        for x in self.env.results().union(self.env.ops()):
            try: ret.update(x.c_compile_args())
            except AbstractFunctionError: pass
        return ret

    def headers(self):
        ret = set()
        for x in self.env.results().union(self.env.ops()):
            try: ret.update(x.c_headers())
            except AbstractFunctionError: pass
        return ret
    
    def libraries(self):
        ret = set()
        for x in self.env.results().union(self.env.ops()):
            try: ret.update(x.c_libraries())
            except AbstractFunctionError: pass
        return ret

    def make_function(self, in_order, out_order):
        nin = len(self.inputs)
        nout = len(self.outputs)
        
        if nin != len(in_order):
            raise TypeError("Wrong number of inputs.")
        if nout != len(out_order):
            raise TypeError("Wrong number of outputs.")
        
        in_storage = []
        out_storage = []

        cthunk_in_args = [None] * nin
        cthunk_out_args = [None] * nout
        
        for result in in_order:
            idx = self.inputs.index(result)
            storage = [None]
            cthunk_in_args[idx] = storage
            in_storage.append(storage)
        for result in out_order:
            idx = self.outputs.index(result)
            storage = [None]
            cthunk_out_args[idx] = storage
            out_storage.append(storage)

        for arg in cthunk_in_args + cthunk_out_args:
            if arg is None:
                raise Exception("The inputs or outputs are underspecified.")

        error_storage = [None, None, None]
        cthunk = self.cthunk_factory(error_storage, cthunk_in_args, cthunk_out_args)
        
        def execute(*args):
            for arg, storage in zip(args, in_storage):
                storage[0] = arg
            failure = cutils.run_cthunk(cthunk)
            if failure:
                raise error_storage[0], error_storage[1] + " " + str(self.find_task(failure - 1))
            return utils.to_return_values([storage[0] for storage in out_storage])

        return execute

    def cthunk_factory(self, error_storage, in_storage, out_storage):

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
            printf("doing cleanup\\n");
            ((%(struct_name)s*)self)->cleanup();
            free(self);
        }
        """ % dict(struct_name = self.struct_name)
        
        instantiate.customize.add_support_code(self.support_code() + self.struct_code + static)
        for arg in self.compile_args():
            instantiate.customize.add_extra_compile_arg(arg)
        for header in self.headers():
            instantiate.customize.add_header(header)
        for lib in self.libraries():
            instantiate.customize.add_library(lib)

        mod.add_function(instantiate)
        mod.compile(location = compile_dir())
        module = __import__("%s" % (module_name), {}, {}, [module_name])

        ret = module.instantiate(error_storage, *(in_storage + out_storage + [orphan._data for orphan in self.orphans]))
        assert sys.getrefcount(ret) == 2 # refcount leak check
        return ret
    



#     def c_thunk_factory(self):
#         self.refresh()
#         d, names, code, struct, converters = self.c_code()

#         cthunk = object()
#         module_name = md5.md5(code).hexdigest()
#         mod = weave.ext_tools.ext_module(module_name)
#         instantiate = weave.ext_tools.ext_function('instantiate',
#                                                    code,
#                                                    names,
#                                                    local_dict = d,
#                                                    global_dict = {},
#                                                    type_converters = converters)
#         instantiate.customize.add_support_code(self.c_support_code() + struct)
#         for arg in self.c_compile_args():
#             instantiate.customize.add_extra_compile_arg(arg)
#         for header in self.c_headers():
#             instantiate.customize.add_header(header)
#         for lib in self.c_libs():
#             instantiate.customize.add_library(lib)
#         #add_library_dir
        
#         #print dir(instantiate.customize)
#         #print instantiate.customize._library_dirs
#         if os.getenv('OMEGA_BLAS_LD_LIBRARY_PATH'):
#             instantiate.customize.add_library_dir(os.getenv('OMEGA_BLAS_LD_LIBRARY_PATH'))

#         mod.add_function(instantiate)
#         mod.compile(location = _compile_dir())
#         module = __import__("%s" % (module_name), {}, {}, [module_name])

#         def creator():
#             return module.instantiate(*[x.data for x in self.inputs + self.outputs])
#         return creator
    
        
    

#     def code_gen(self, reuse_storage = True):
        
#         env = self.env
#         op_order = env.toposort()
        
#         to_extract = env.inputs.union(env.orphans())
#         to_sync = env.outputs
#         temporaries = env.results().difference(to_extract).difference(to_sync)

#         symbol = {}
        
#         init_tasks = []
#         tasks = []

#         init_blocks = []
#         blocks = []

#         failure_var = "__failure"
#         id = 0

#         sub = dict(failure_var = failure_var)

#         on_stack = [result for result in temporaries.union(to_sync) if not reuse_storage or result.c_is_simple()]
        
#         for result_set, type in [[to_extract, 'input'],
#                                  [to_sync, 'output'],
#                                  [temporaries, 'temporary']]:
#             for result in result_set:
#                 builder, block = struct_result_codeblocks(result, type, id, symbol, sub, on_stack)

#                 init_tasks.append((result, 'init'))
#                 init_blocks.append(builder)

#                 tasks.append((result, 'get'))
#                 blocks.append(block)

#                 id += 2

#         for op in op_order:

#             ivnames, ovnames = op.c_var_names()
#             sub = dict(failure_var = failure_var)
#             for result, vname in zip(op.inputs + op.outputs, ivnames + ovnames):
#                 sub[vname] = symbol[result]

#             # c_validate_update
#             try: validate_behavior = op.c_validate_update()
#             except AbstractFunctionError:
#                 validate_behavior = ""

#             try: validate_behavior = op.c_validate_update_cleanup()
#             except AbstractFunctionError:
#                 validate_cleanup = ""

#             sub['id'] = id
#             blocks.append(CodeBlock("", validate_behavior, validate_cleanup, sub))
#             tasks.append((op, 'validate_update'))
#             id += 1

#             # c_code
#             behavior = op.c_code() # this one must be implemented!

#             try: cleanup = op.c_code_cleanup()
#             except AbstractFunctionError:
#                 cleanup = ""
            
#             sub['id'] = id
#             blocks.append(CodeBlock("", behavior, cleanup, sub))
#             tasks.append((op, 'code'))
#             id += 1

#         args = []
#         in_arg_order = []
#         for result in list(to_extract):
#             in_arg_order.append(result)
#             args.append("storage_%s" % symbol[result])
#         out_arg_order = []
#         for result in to_sync:
#             out_arg_order.append(result)
#             args.append("storage_%s" % symbol[result])
#         struct_code = struct_gen(args, init_blocks, blocks, dict(failure_var = failure_var))

#         hash = md5.md5(struct_code).hexdigest()
#         struct_name = 'compiled_op_%s' % hash
#         struct_code %= dict(name = struct_name)

#         self.struct_code = struct_code
#         self.struct_name = struct_name
#         self.hash = hash
#         self.args = args
#         self.inputs = in_arg_order
#         self.outputs = out_arg_order
#         self.r2symbol = symbol
#         self.init_blocks = init_blocks
#         self.init_tasks = init_tasks
#         self.blocks = blocks
#         self.tasks = tasks
        
#         return struct_code
        
























    
        
#     def extract_sync(self, to_extract, to_sync, to_cleanup):
#         pass

#     def code_gen(self):
#         env = self.env
#         order = env.toposort()
#         to_extract = env.inputs.union(env.outputs).union(env.orphans())
#         head = ""
#         tail = ""
        
#         label_id = 0        
#         name_id = 0
#         result_names = {}
#         for result in env.results():
#             name = "__v_%i" % name_id
#             result_names[result] = name
#             name_id += 1
        
#         for result in to_extract:
#             head += """
#             {
#                 %(extract)s
#             """
#             tail = """
#             __label_%(label_id)s:
#                 %(sync)s
#             }
#             """ + tail
#             name = result_names[result]
#             type = result.c_type()
#             head %= dict(extract = result.c_extract())
#             head %= dict(name = name,
#                          type = type,
#                          fail = "{goto __label_%i;}" % label_id)
#             tail %= dict(sync = result.c_sync(),
#                          label_id = label_id)
#             tail %= dict(name = name,
#                          type = type)
#             label_id += 1

#         for op in order:
#             inames, onames = op.c_var_names()
            
        
#         return head + tail



# def struct_result_codeblocks(result, type, id, symbol_table, sub, on_stack):

#     if type == 'output':
#         sync = get_c_sync(result)
#     else:
#         sync = ""

#     if type == 'input':
#         struct_declare = ""
#         run_declare = result.c_declare()

#         struct_behavior = ""
#         run_behavior = get_c_extract(result)

#         struct_cleanup = ""
#         run_cleanup = get_c_cleanup(result)

#     else:
#         if result in on_stack:
#             struct_declare = ""
#             run_declare = result.c_declare()

#             struct_behavior = ""
#             run_behavior = result.c_init()

#             struct_cleanup = ""
#             run_cleanup = sync + get_c_cleanup(result)

#         else:
#             struct_declare = result.c_declare()
#             run_declare = ""

#             struct_behavior = result.c_init()
#             run_behavior = ""

#             struct_cleanup = get_c_cleanup(result)
#             run_cleanup = sync

#     name = "V%i" % id
#     symbol_table[result] = name
#     sub = copy(sub)
#     sub['name'] = name
#     sub['id'] = id
#     struct_builder = CodeBlock(struct_declare, struct_behavior, struct_cleanup, sub)
#     sub['id'] = id + 1
#     block = CodeBlock(run_declare, run_behavior, run_cleanup, sub)

#     return struct_builder, block

