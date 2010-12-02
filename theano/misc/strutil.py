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
