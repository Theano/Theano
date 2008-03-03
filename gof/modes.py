
import utils
from op import Op

__all__ = ['ModalConstructor',
           'add_modal_members',
           'ModalWrapper',
           'BuildMode',
           'EvalMode',
           'BuildEvalMode',
           'make_constructors',
           ]

class ModalConstructor:

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        modal_wrapper = None
        fn_args = []
        for arg in args:
            if isinstance(arg, ModalWrapper):
                if modal_wrapper is None:
                    modal_wrapper = arg.__class__
                else:
                    if not isinstance(arg, modal_wrapper):
                        raise TypeError("Inconsistent modes.")
                fn_args.append(arg.r)
            else:
                fn_args.append(arg)
        op = self.fn(*fn_args)
        if modal_wrapper:
            modal_wrapper.filter(op)
        if len(op.outputs) == 1:
            return modal_wrapper(op.outputs[0])
        else:
            return [modal_wrapper(output) for output in op.outputs]


def add_modal_members(cls, *members):
    def fn(member):
        def ret(self, *args):
            constructor = ModalConstructor(getattr(self.r.__class__, member))
            return constructor(self, *args)
        return ret
    for member in members:
        setattr(cls, member, fn(member))


class ModalWrapper:

    def __init__(self, r):
        self.r = r

    @classmethod
    def filter(cls, op):
        raise AbstractFunctionError()

members1 = 'add sub mul div pow floordiv mod pow lshift rshift and or xor'.split(' ')
members = []
members += ["__%s__" % x for x in members1 + 'neg invert'.split(' ')]
members += ["__r%s__" % x for x in members1]
add_modal_members(ModalWrapper, *members)



class BuildMode(ModalWrapper):
    @classmethod
    def filter(cls, op):
        pass

class EvalMode(ModalWrapper):
    @classmethod
    def filter(cls, op):
        op.perform()
        for output in op.outputs:
            output._role = None

class BuildEvalMode(ModalWrapper):
    @classmethod
    def filter(cls, op):
        op.perform()


def _is_op(x):
    try: return issubclass(x, Op)
    except: return False

def make_constructors(source,
                      dest = None,
                      name_filter = utils.camelcase_to_separated,
                      candidate_filter = _is_op):
    if dest is None:
        dest = source
    for symbol, value in source.items():
        if candidate_filter(value):
            dest[name_filter(symbol)] = ModalConstructor(value)
    return dest


