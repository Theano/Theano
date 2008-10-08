
import os
from docutils import nodes
import epydoc.docwriter.xlink as xlink

def role_fn(name, rawtext, text, lineno, inliner,
            options={}, content=[]):
    node = nodes.reference(rawtext, text, refuri = "http://pylearn.org/theano/wiki/%s" % text)
    return [node], []


def setup(app):
    help(xlink)
    #role = xlink.create_api_role('api', True)
    #print role

    xlink.register_api('api', xlink.DocUrlGenerator())
    xlink.set_api_file('api', os.path.join(app.outdir, 'api', 'api-objects.txt'))
    xlink.set_api_root('api', os.path.join(app.outdir, 'api', ''))
    xlink.create_api_role('api', True)
    
    app.add_role("wiki", role_fn)
