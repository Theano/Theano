
import sys
import re
import os
from docutils import nodes, utils
from docutils.parsers.rst import roles
import epydoc.docwriter.xlink as xlink

def role_fn(name, rawtext, text, lineno, inliner,
            options={}, content=[]):
    node = nodes.reference(rawtext, text, refuri = "http://pylearn.org/theano/wiki/%s" % text)
    return [node], []


_TARGET_RE = re.compile(r'^(.*?)\s*<(?:URI:|URL:)?([^<>]+)>$') 
def create_api_role(name, problematic):
    """
    Create and register a new role to create links for an API documentation.

    Create a role called `name`, which will use the URL resolver registered as
    ``name`` in `api_register` to create a link for an object.

    :Parameters:
      `name` : `str`
        name of the role to create.
      `problematic` : `bool`
        if True, the registered role will create problematic nodes in
        case of failed references. If False, a warning will be raised
        anyway, but the output will appear as an ordinary literal.
    """
    def resolve_api_name(n, rawtext, text, lineno, inliner,
                options={}, content=[]):

        # Check if there's separate text & targets 
        m = _TARGET_RE.match(text) 
        if m: text, target = m.groups() 
        else: target = text 
        
        # node in monotype font
        text = utils.unescape(text)
        node = nodes.literal(rawtext, text, **options)

        # Get the resolver from the register and create an url from it.
        try:
            url = xlink.api_register[name].get_url(target)
        except IndexError, exc:
            msg = inliner.reporter.warning(str(exc), line=lineno)
            if problematic:
                prb = inliner.problematic(rawtext, text, msg)
                return [prb], [msg]
            else:
                return [node], []

        if url is not None:
            node = nodes.reference(rawtext, '', node, refuri=url, **options)
        return [node], []

    roles.register_local_role(name, resolve_api_name)


def setup(app):

    try:
        xlink.set_api_file('api', os.path.join(app.outdir, 'api', 'api-objects.txt'))
        xlink.set_api_root('api', os.path.join('..', 'api', ''))
        #xlink.create_api_role('api', True)
        create_api_role('api', True)
    except IOError:
        print >>sys.stderr, 'WARNING: Could not find api file! API links will not work.'

    app.add_role("wiki", role_fn)
