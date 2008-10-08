
import sys
import os
import inspect

from epydoc import docintrospecter 
from epydoc.apidoc import RoutineDoc

def Op_to_RoutineDoc(op, routine_doc, module_name=None):
    routine_doc.specialize_to(RoutineDoc)

    #NB: this code is lifted from
    # /u/bergstrj/pub/prefix/x86_64-unknown-linux-gnu-Fedora_release_7__Moonshine_/lib/python2.5/site-packages/epydoc
    # /u/bergstrj/pub/prefix/x86_64-unknown-linux-gnu-Fedora_release_7__Moonshine_/lib/python2.5/site-packages/epydoc/docintrospecter.py


    # op should be an op instance
    assert hasattr(op, 'perform')

    # Record the function's docstring.
    routine_doc.docstring = getattr(op, '__doc__', '')

    # Record the function's signature.
    func = op.__epydoc_asRoutine
    if isinstance(func, type(Op_to_RoutineDoc)):
        (args, vararg, kwarg, defaults) = inspect.getargspec(func)

        # Add the arguments.
        routine_doc.posargs = args
        routine_doc.vararg = vararg
        routine_doc.kwarg = kwarg

        # Set default values for positional arguments.
        routine_doc.posarg_defaults = [None]*len(args)

        # Set the routine's line number.
        if hasattr(func, 'func_code'):
            routine_doc.lineno = func.func_code.co_firstlineno
    else:
        # [XX] I should probably use UNKNOWN here??
        # dvarrazzo: if '...' is to be changed, also check that
        # `docstringparser.process_arg_field()` works correctly.
        # See SF bug #1556024.
        routine_doc.posargs = ['...']
        routine_doc.posarg_defaults = [None]
        routine_doc.kwarg = None
        routine_doc.vararg = None

    return routine_doc

docintrospecter.register_introspecter(
    lambda value: getattr(value, '__epydoc_asRoutine', False), 
    Op_to_RoutineDoc, 
    priority=-1)



if __name__ == '__main__':

    throot = "/".join(sys.path[0].split("/")[:-1])

    import gen_oplist
    print 'Generating oplist...'
    gen_oplist.print_file(open('%s/doc/doc/oplist.txt' % throot, 'w'))
    print 'oplist done!'

    import gen_typelist
    print 'Generating typelist...'
    gen_typelist.print_file(open('%s/doc/doc/typelist.txt' % throot, 'w'))
    print 'typelist done!'

    os.chdir(throot)

    def mkdir(path):
        try:
            os.mkdir(path)
        except OSError:
            pass

    mkdir("html")
    mkdir("html/doc")
    mkdir("html/api")

    if len(sys.argv) == 1 or sys.argv[1] != 'rst':
        from epydoc.cli import cli
        sys.path[0:0] = os.path.realpath('.')
        sys.argv[:] = ['', '--config', 'doc/api/epydoc.conf', '-o', 'html/api']
        cli()
#        os.system("epydoc --config doc/api/epydoc.conf -o html/api")

    if len(sys.argv) == 1 or sys.argv[1] != 'epydoc':
        import sphinx
        sys.path[0:0] = [os.path.realpath('doc')]
        sphinx.main(['', '-E', 'doc', 'html'])


