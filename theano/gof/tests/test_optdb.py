from __future__ import absolute_import, print_function, division
from unittest import TestCase

from theano.compat import exc_message
from theano.gof.optdb import opt, DB


class Test_DB(TestCase):

    def test_0(self):

        class Opt(opt.Optimizer):  # inheritance buys __hash__
            name = 'blah'

        db = DB()
        db.register('a', Opt())

        db.register('b', Opt())

        db.register('c', Opt(), 'z', 'asdf')

        self.assertTrue('a' in db)
        self.assertTrue('b' in db)
        self.assertTrue('c' in db)

        try:
            db.register('c', Opt())  # name taken
            self.fail()
        except ValueError as e:
            if exc_message(e).startswith("The name"):
                pass
            else:
                raise
        except Exception:
            self.fail()

        try:
            db.register('z', Opt())  # name collides with tag
            self.fail()
        except ValueError as e:
            if exc_message(e).startswith("The name"):
                pass
            else:
                raise
        except Exception:
            self.fail()

        try:
            db.register('u', Opt(), 'b')  # name new but tag collides with name
            self.fail()
        except ValueError as e:
            if exc_message(e).startswith("The tag"):
                pass
            else:
                raise
        except Exception:
            self.fail()
