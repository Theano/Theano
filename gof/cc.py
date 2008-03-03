
from link import Linker


class CLinker(Linker):

    def __init__(self, env):
        self.env = env

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


