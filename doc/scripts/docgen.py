
import sys
import os
import shutil
import inspect

from epydoc import docintrospecter
from epydoc.apidoc import RoutineDoc


def Op_to_RoutineDoc(op, routine_doc, module_name=None):
    routine_doc.specialize_to(RoutineDoc)

    #NB: this code is lifted from epydoc/docintrospecter.py

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
        routine_doc.posarg_defaults = [None] * len(args)

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


import getopt
from collections import defaultdict

if __name__ == '__main__':
    # Equivalent of sys.path[0]/../..
    throot = os.path.abspath(
        os.path.join(sys.path[0], os.pardir, os.pardir))

    options = defaultdict(bool)
    options.update(dict([x, y or True] for x, y in
        getopt.getopt(sys.argv[1:],
                      'o:',
                      ['epydoc', 'rst', 'help', 'nopdf', 'cache'])[0]))
    if options['--help']:
        print 'Usage: %s [OPTIONS]' % sys.argv[0]
        print '  -o <dir>: output the html files in the specified dir'
        print '  --cache: use the doctree cache'
        print '  --rst: only compile the doc (requires sphinx)'
        print '  --nopdf: do not produce a PDF file from the doc, only HTML'
        print '  --epydoc: only compile the api documentation',
        print '(requires epydoc)'
        print '  --help: this help'
        sys.exit(0)

    if not (options['--epydoc'] or options['--rst']):
        # Default is now rst
        options['--rst'] = True

    def mkdir(path):
        try:
            os.mkdir(path)
        except OSError:
            pass

    outdir = options['-o'] or (throot + '/html')
    mkdir(outdir)
    os.chdir(outdir)

    # Make sure the appropriate 'theano' directory is in the PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    pythonpath = os.pathsep.join([throot, pythonpath])
    sys.path[0:0] = [throot]  # We must not use os.environ.

    if options['--all'] or options['--epydoc']:
        mkdir("api")
        sys.path[0:0] = [throot]

        #Generate HTML doc

        ## This causes problems with the subsequent generation of sphinx doc
        #from epydoc.cli import cli
        #sys.argv[:] = ['', '--config', '%s/doc/api/epydoc.conf' % throot,
        #               '-o', 'api']
        #cli()
        ## So we use this instead
        os.system("epydoc --config %s/doc/api/epydoc.conf -o api" % throot)

        # Generate PDF doc
        # TODO

    if options['--all'] or options['--rst']:
        mkdir("doc")
        sys.path[0:0] = [os.path.join(throot, 'doc')]
        def call_sphinx(builder, workdir, extraopts=None):
            import sphinx
            if extraopts is None:
                extraopts = []
            if not options['--cache']:
                extraopts.append('-E')
            sphinx.main(['', '-b', builder] + extraopts +
                        [os.path.join(throot, 'doc'), workdir])
        call_sphinx('html', '.')

        if not options['--nopdf']:
            # Generate latex file in a temp directory
            import tempfile
            workdir = tempfile.mkdtemp()
            call_sphinx('latex', workdir)
            # Compile to PDF
            os.chdir(workdir)
            os.system('make')
            try:
                shutil.copy(os.path.join(workdir, 'theano.pdf'), outdir)
                os.chdir(outdir)
                shutil.rmtree(workdir)
            except OSError, e:
                print 'OSError:', e
            except IOError, e:
                print 'IOError:', e
