import cPickle, logging, sys

_logger=logging.getLogger("theano.gof.callcache")

def warning(*args):
    sys.stderr.write('WARNING:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.warning(' '.join(str(a) for a in args))
def error(*args):
    sys.stderr.write('ERROR:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.error(' '.join(str(a) for a in args))
def info(*args):
    sys.stderr.write('INFO:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.info(' '.join(str(a) for a in args))
def debug(*args):
    sys.stderr.write('DEBUG:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.debug(' '.join(str(a) for a in args))


class CallCache(object):
    def __init__(self, filename=None):
        self.filename = filename
        try:
            if filename is None:
                raise IOError('bad filename') #just goes to except 
            f = file(filename, 'r')
            self.cache = cPickle.load(f)
            f.close()
        except IOError:
            self.cache = {}

    def persist(self, filename=None):
        if filename is None:
          filename = self.filename

        #backport
        #filename = self.filename if filename is None else filename
        f = file(filename, 'w')
        cPickle.dump(self.cache, f)
        f.close()

    def call(self, fn, args=(), key=None):
        if key is None:
          key = (fn, tuple(args))

        #backport
        #key = (fn, tuple(args)) if key is None else key
        if key not in self.cache:
            debug('cache miss', len(self.cache))
            self.cache[key] = fn(*args)
        else:
            debug('cache hit', len(self.cache))
        return self.cache[key]

    def __del__(self):
        try:
            if self.filename:
                self.persist()
        except Exception, e:
            _logging.error('persist failed', self.filename, e)

