__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from theano.compile import Mode
import theano
from pylearn2.utils import hex_digest

class MismatchError(Exception):
    """
    Raised by Record.handle_line when the
    current execution doesn't match the replay
    of a record.
    """

class Record(object):
    def __init__(self, file_object=None, file_path=None, replay=False):

        assert file_object is not None or file_path is not None

        if replay and file_object is None:
            self.f = open(file_path, 'r')
        elif (not replay) and file_object is None:
            self.f = open(file_path, 'w')
        else:
            self.f = file_object

        self.__dict__.update(locals())

    def handle_line(self, line):
        assert line.endswith('\n')
        assert line[:-2].find('\n') == -1
        if self.replay:
            old_line = self.f.readline()
            if old_line != line:
                msg = 'Replay detected mismatch.\n'
                msg += ' I wanted to write:\n'
                if len(line) > 100:
                    msg += line[0:100]+'...'
                else:
                    msg += line
                msg += '\nwhen previous job wrote:\n'
                if len(old_line) > 100:
                    msg += old_line[0:100]+'...'
                else:
                    msg += old_line
                raise MismatchError(msg)
        else:
            self.f.write(line)

class RecordMode(Mode):
    """
    Records all computations done with a function in a file at output_path
    Prints the index of each apply node and md5 digests of the numpy ndarrays
    it receives as inputs and produces as outputs.
    """

    def set_record(self, record):
        self.record = record
        self.known_fgraphs = set([])

    def __init__(self, record = None, **kwargs):
        """
        Takes either a Record object or the keyword arguments to make one.
        """

        if record is None:
            record = Record(**kwargs)
        else:
            assert len(kwargs.keys()) == 0

        self.set_record(record)


        def handle_line(line, i, node, fn):
            try:
                self.record.handle_line(line)
            except MismatchError, e:
                print 'Got this MismatchError:'
                print e
                print 'while processing node i='+str(i)+':'
                print 'str(node):',str(node)
                print 'Symbolic inputs: '
                for elem in node.inputs:
                    print theano.printing.min_informative_str(elem)
                print 'str(output) of outputs: '
                for elem in fn.outputs:
                    assert isinstance(elem, list)
                    elem, = elem
                    print str(elem)
                print 'function name: '+node.fgraph.name
                raise MismatchError("Non-determinism detected by WrapLinker")

        def callback(i, node, fn):

            fgraph = node.fgraph

            if fgraph.name is None:
                raise ValueError("Un-named functions are not allowed with RecordMode, "
                        "because they make it impossible to tell if the same function is "
                        "running during the playback.")

            if fgraph not in self.known_fgraphs:
                assert not any([elem.name == fgraph.name for elem in self.known_fgraphs])
                self.known_fgraphs.add(fgraph)
                num_app = len(fgraph.apply_nodes)
                line = 'Function '+fgraph.name+' has '+str(num_app)+' apply nodes.\n'
                handle_line(line, i, node, fn)

            line = 'Function name: '+fgraph.name + '\n'
            handle_line(line, i, node, fn)
            line = 'Node '+str(i)+':'+str(node)+'\n'
            handle_line(line, i, node, fn)
            assert all([isinstance(x, list) and len(x) == 1 for x in fn.inputs])
            def digest(x):
                x = x[0]
                return hex_digest(x)
            inputs_digest = ' '.join([digest(x) for x in fn.inputs])
            line = 'Inputs: ' + inputs_digest + '\n'
            handle_line(line, i, node, fn)
            fn()
            outputs_digest = ' '.join([digest(x) for x in fn.outputs])
            line = 'Outputs: ' + outputs_digest + '\n'
            handle_line(line, i, node, fn)

        #linker = theano.gof.OpWiseCLinker()
        linker = theano.gof.vm.VM_Linker(use_cloop=True)

        wrap_linker = theano.gof.WrapLinkerMany([linker], [callback])
        super(RecordMode, self).__init__(wrap_linker, optimizer='fast_run')
