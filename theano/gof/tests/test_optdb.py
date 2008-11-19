from theano.gof.optdb import *
from unittest import TestCase

class Test_DB(TestCase):

    def test_0(self):

        class Opt(opt.Optimizer):  #inheritance buys __hash__
            name = 'blah'

        db = DB()
        db.register('a', Opt())

        db.register('b', Opt())

        db.register('c', Opt(), 'z', 'asdf')

        try:
            db.register('c', Opt()) #name taken
            self.fail()
        except ValueError, e:
            if e[0].startswith("The name"):
                pass
            else:
                raise
        except:
            self.fail()

        try:
            db.register('z', Opt()) #name collides with tag
            self.fail()
        except ValueError, e:
            if e[0].startswith("The name"):
                pass
            else:
                raise
        except:
            self.fail()

        try:
            db.register('u', Opt(), 'b') #name new but tag collides with name
            self.fail()
        except ValueError, e:
            if e[0].startswith("The tag"):
                pass
            else:
                raise
        except:
            self.fail()


