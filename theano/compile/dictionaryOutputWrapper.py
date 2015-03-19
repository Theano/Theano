'''
Generates a wrapper around theano functions that allows the user to receive outputs in a dictionary.  
'''

def createFunctionReturningDictionary(args, kwargs, fn, keys):

    outputLst = fn(*args, **kwargs)

    outputDict = {}

    for i in range(0, len(keys)):
        outputDict[keys[i]] = outputLst[i]

    return outputDict


