
import utils
import traceback
from op import Op

__all__ = ['ModalConstructor',
           'add_modal_members',
           'build',
           'eval',
           'build_eval',
           'make_constructors',
           ]

class ModalConstructor:

    def __init__(self, fn):
        self.fn = fn
    
    def __call__(self, *args):
        modal_wrapper = None
        fn_args = []
        for arg in args:
            mode = getattr(arg, '__mode__', False)
            if mode:
                if modal_wrapper is None:
                    modal_wrapper = mode
                else:
                    if mode != modal_wrapper:
                        raise TypeError("Inconsistent modes.")
            fn_args.append(arg)
        op = self.fn(*fn_args)
        if modal_wrapper:
            modal_wrapper(op)
        for output in op.outputs:
            output.__mode__ = modal_wrapper
        if len(op.outputs) == 1:
            return op.outputs[0]
        else:
            return op.outputs


def add_modal_members(cls, *members):
    def fn(member):
        def ret(self, *args):
            constructor = ModalConstructor(getattr(self.r.__class__, member))
            return constructor(self, *args)
        return ret
    for member in members:
        setattr(cls, member, fn(member))


def attach_trace(op):
    stack = traceback.extract_stack()[:-3]
    op.trace = stack

def build_mode(op):
    attach_trace(op)

def eval_mode(op):
    attach_trace(op)
    op.perform()
    for output in op.outputs:
        output._role = None

def build_eval_mode(op):
    attach_trace(op)
    op.perform()


def mode_setter(mode):
    def f(r):
        r.__mode__ = mode
        return r
    return f

build = mode_setter(build_mode)
eval = mode_setter(eval_mode)
build_eval = mode_setter(build_eval_mode)


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


