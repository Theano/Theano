from theano.d3viz.d3viz import d3viz, d3write

has_requirements = False
try:
    import pydot as pd
    if pd.find_graphviz():
        has_requirements = True
except ImportError:
    pass
