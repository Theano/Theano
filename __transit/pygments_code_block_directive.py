#!/usr/bin/python

# :Author: a Pygments author|contributor; Felix Wiemann; Guenter Milde
# :Date: $Date: 2007-06-13 12:20:42 +0200 (Wed, 13 Jun 2007) $
# :Copyright: This module has been placed in the public domain.
# 
# This is a merge of `Using Pygments in ReST documents`_ from the pygments_
# documentation, and a `proof of concept`_ by Felix Wiemann.
# 
# ========== ===========================================================
# 2007-06-01 Removed redundancy from class values.
# 2007-06-04 Merge of successive tokens of same type
#            (code taken from pygments.formatters.others).
# 2007-06-05 Separate docutils formatter script
#            Use pygments' CSS class names (like the html formatter)
#            allowing the use of pygments-produced style sheets.
# 2007-06-07 Merge in the formatting of the parsed tokens
#            (misnamed as docutils_formatter) as class DocutilsInterface
# 2007-06-08 Failsave implementation (fallback to a standard literal block 
#            if pygments not found)
# ========== ===========================================================
# 
# ::

"""Define and register a code-block directive using pygments
"""

# Requirements
# ------------
# ::

from docutils import nodes
from docutils.parsers.rst import directives

try:
    import pygments
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters.html import _get_ttype_class
    from pygments.styles import get_style_by_name
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter

    # Customisation
    # -------------
    # 
    # Do not insert inline nodes for the following tokens.
    # (You could add e.g. Token.Punctuation like ``['', 'p']``.) ::

    unstyled_tokens = ['']

    # DocutilsInterface
    # -----------------
    # 
    # This interface class combines code from
    # pygments.formatters.html and pygments.formatters.others.
    # 
    # It does not require anything of docutils and could also become a part of
    # pygments::

    class DocutilsInterface(object):
        """Parse `code` string and yield "classified" tokens.
        
        Arguments
        
          code     -- string of source code to parse
          language -- formal language the code is written in.
        
        Merge subsequent tokens of the same token-type. 
        
        Yields the tokens as ``(ttype_class, value)`` tuples, 
        where ttype_class is taken from pygments.token.STANDARD_TYPES and 
        corresponds to the class argument used in pygments html output.

        """

        def __init__(self, code, language):
            self.code = code
            self.language = language
            
        def lex(self):
            # Get lexer for language (use text as fallback)
            try:
                lexer = get_lexer_by_name(self.language)
            except ValueError:
                # info: "no pygments lexer for %s, using 'text'"%self.language
                lexer = get_lexer_by_name('text')
            return pygments.lex(self.code, lexer)
            
                
        def join(self, tokens):
            """join subsequent tokens of same token-type
            """
            tokens = iter(tokens)
            (lasttype, lastval) = tokens.next()
            for ttype, value in tokens:
                if ttype is lasttype:
                    lastval += value
                else:
                    yield(lasttype, lastval)
                    (lasttype, lastval) = (ttype, value)
            yield(lasttype, lastval)

        def __iter__(self):
            """parse code string and yield "clasified" tokens
            """
            try:
                tokens = self.lex()
            except IOError:
                print "INFO: Pygments lexer not found, using fallback"
                # TODO: write message to INFO 
                yield ('', self.code)
                return

            for ttype, value in self.join(tokens):
                yield (_get_ttype_class(ttype), value)



    # code_block_directive
    # --------------------
    # ::

    def code_block_directive(name, arguments, options, content, lineno,
                           content_offset, block_text, state, state_machine):
        """parse and classify content of a code_block
        """
        language = arguments[0]
        # create a literal block element and set class argument
        if 0:
            code_block = nodes.literal_block(classes=["code-block", language])
            code_block += nodes.raw('<b>hello</b> one', 'hello two')
        else:
            code_block = nodes.literal_block(classes=["code-block", language])
            
            # parse content with pygments and add to code_block element
            for cls, value in DocutilsInterface(u'\n'.join(content), language):
                if cls in unstyled_tokens:
                    # insert as Text to decrease the verbosity of the output.
                    code_block += nodes.Text(value, value)
                else:
                    code_block += nodes.inline(value, value, classes=[cls])

            if 0:
                v = highlight(u'\n'.join(content), PythonLexer(), 
                        HtmlFormatter(style='colorful', full=True, cssfile='blah.css'))
                print help(nodes.Inline)

        return [code_block]


    # Register Directive
    # ------------------
    # ::

    code_block_directive.arguments = (1, 0, 1)
    code_block_directive.content = 1
    directives.register_directive('code-block', code_block_directive)

    # .. _doctutils: http://docutils.sf.net/
    # .. _pygments: http://pygments.org/
    # .. _Using Pygments in ReST documents: http://pygments.org/docs/rstdirective/
    # .. _proof of concept:
    #      http://article.gmane.org/gmane.text.docutils.user/3689
    # 
    # Test output
    # -----------
    # 
    # If called from the command line, call the docutils publisher to render the
    # input::

except ImportError:
    print >> sys.stderr, "Failed to import pygments"
    pass



if __name__ == '__main__':
    from docutils.core import publish_cmdline, default_description
    description = "code-block directive test output" + default_description
    try:
        import locale
        locale.setlocale(locale.LC_ALL, '')
    except:
        pass
    # Uncomment the desired output format:
    publish_cmdline(writer_name='pseudoxml', description=description)
    # publish_cmdline(writer_name='xml', description=description)
    # publish_cmdline(writer_name='html', description=description)
    # publish_cmdline(writer_name='latex', description=description)
    # publish_cmdline(writer_name='newlatex2e', description=description)
    


