#!/usr/bin/env python

"""Creates reference files for testing d3viz module.
Used in tests/test_d3viz.py.

Author: Christof Angermueller <cangermueller@gmail.com>
"""

import theano as th
import theano.d3viz as d3v
from theano.d3viz.tests import models


model = models.Mlp()
f = th.function(model.inputs, model.outputs)
d3v.d3viz(f, 'mlp/index.html')

model = models.Ofg()
f = th.function(model.inputs, model.outputs)
d3v.d3viz(f, 'ofg/index.html')

model = models.OfgNested()
f = th.function(model.inputs, model.outputs)
d3v.d3viz(f, 'ofg_nested/index.html')
