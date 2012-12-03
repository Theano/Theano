__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

from datetime import datetime

def disturb_mem():
    # Allocate a time-dependent amount of objects to increase
    # chances of subsequently objects' ids changing from run
    # to run. This is useful for exposing issues that cause
    # non-deterministic behavior due to dependence on memory
    # addresses, like iterating over a dict or a set.
    global l
    now = datetime.now()
    ms = now.microsecond
    ms = int(ms)
    n = ms % 1000
    m = ms / 1000
    l = [[0]*m for i in xrange(n)]
