from theano.d3viz.d3viz import d3viz, d3write

has_requirements = True
try:
    import pydot
except ImportError:
    has_requirements = False
