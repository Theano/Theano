from theano.d3viz.d3viz import d3viz, d3write

has_requirements = False
try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pd
except ImportError:
    # fall back on pydot if necessary
    import pydot as pd
if pd.find_graphviz():
    has_requirements = True
