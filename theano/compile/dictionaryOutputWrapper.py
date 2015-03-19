'''
Supports user generating theano function where the outputs are a dictionary
'''

def createFunctionReturningDictionary(args, fn, keys):
    outputLst = fn(*args)

    outputDict = {}

    for i in range(0, len(keys)):
        outputDict[keys[i]] = outputLst[i]

    return outputDict

