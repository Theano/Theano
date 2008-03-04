
from link import Linker
from copy import copy


class CodeBlock:

    def __init__(self, declare, behavior, cleanup, sub):
        self.declare = declare % sub
        behavior_sub = copy(sub)
        behavior_sub['fail'] = "{goto __label_%(id)i}" % sub
        self.behavior = behavior % behavior_sub
        self.cleanup = ("__label_%(id)i:\n" + cleanup) % sub


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
    
    for block in blocks:
        struct_decl += block.declare
        struct_init_head = struct_init_head + ("\n{\n%s" % block.behavior)
        struct_init_tail = ("%s\n}\n" % block.cleanup) + struct_init_tail
        struct_cleanup += block.cleanup

    behavior = code_gen(blocks)

    args = ", ".join(["PyObject* %s" % arg for arg in args])

    struct_code = """
    struct __struct_%%(id)s {
        %(struct_decl)s
        
        __struct_%%(id)s(void) {}
        ~__struct_%%(id)s(void) {
            cleanup();
        }
        void init(%(args)s) {
            %(struct_init_head)s
            return;
            %(struct_init_tail)s
        }
        void cleanup(void) {
            %(struct_cleanup)s
        }
        void run(void) {
            %(behavior)s
        }
    };
    """ % locals()

    return struct_code


class CLinker(Linker):

    def __init__(self, env):
        self.env = env

    def extract_sync(self, to_extract, to_sync, to_cleanup):
        pass

    def code_gen(self):
        env = self.env
        order = env.toposort()
        to_extract = env.inputs.union(env.outputs).union(env.orphans())
        head = ""
        tail = ""
        label_id = 0
        
        name_id = 0
        result_names = {}
        for result in env.results():
            name = "__v_%i" % name_id
            result_names[result] = name
            name_id += 1
        
        for result in to_extract:
            head += """
            {
                %(extract)s
            """
            tail = """
            __label_%(label_id)s:
                %(sync)s
            }
            """ + tail
            name = result_names[result]
            type = result.c_type()
            head %= dict(extract = result.c_extract())
            head %= dict(name = name,
                         type = type,
                         fail = "{goto __label_%i;}" % label_id)
            tail %= dict(sync = result.c_sync(),
                         label_id = label_id)
            tail %= dict(name = name,
                         type = type)
            label_id += 1

        for op in order:
            inames, onames = op.c_var_names()
            
        
        return head + tail


