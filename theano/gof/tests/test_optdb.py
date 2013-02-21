from unittest import TestCase
from six import PY3
if PY3:
    # In python 3.x, when an exception is reraised it saves original
    # exception in its args, therefore in order to find the actual
    # message, we need to unpack arguments recurcively.
    def exc_message(e):
        msg = e.args[0]
        if isinstance(msg, Exception):
            return exc_message(msg)
        return msg
else:
    def exc_message(e):
        return e[0]

from theano.gof.optdb import opt, DB


class Test_DB(TestCase):

    def test_0(self):

        class Opt(opt.Optimizer):  # inheritance buys __hash__
            name = 'blah'

        db = DB()
        db.register('a', Opt())

        db.register('b', Opt())

        db.register('c', Opt(), 'z', 'asdf')

        try:
            db.register('c', Opt())  # name taken
            self.fail()
        except ValueError, e:
            if exc_message(e).startswith("The name"):
                pass
            else:
                raise
        except Exception:
            self.fail()

        try:
            db.register('z', Opt())  # name collides with tag
            self.fail()
        except ValueError, e:
            if exc_message(e).startswith("The name"):
                pass
            else:
                raise
        except Exception:
            self.fail()

        try:
            db.register('u', Opt(), 'b')  # name new but tag collides with name
            self.fail()
        except ValueError, e:
            if exc_message(e).startswith("The tag"):
                pass
            else:
                raise
        except Exception:
            self.fail()
