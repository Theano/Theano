def renderString(string, dict):
    try:
        finalCode = string % dict
    except Exception , E:
        #print 'could not render C code due to exception with message "'+str(E)+'", trying to find out why...'
        i = 0
        while i <= len(string):
            try:
                finalCode = string[0:i] % dict
            except Exception, F:
                if str(F) == str(E):
                    raise Exception(string[0:i]+"<<<< caused exception "+str(F))
            i+=1
        assert False
    return finalCode
#

def pretty_format(string):
    lines = string.split('\n')

    lines = [ strip_leading_white_space(line) for line in lines ]

    indent = 0
    for i in xrange(len(lines)):
        indent -= lines[i].count('}')
        if indent < 0:
            indent = 0
        #
        lines[i] = ('    '*indent) + lines[i]
        indent += lines[i].count('{')
    #


    rval = '\n'.join(lines)

    return rval
#

def strip_leading_white_space(line):
    while len(line) >0 and (line[0]==' ' or line[0]=='\t'):
        line = line[1:]
    #
    return line
#
