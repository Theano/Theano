import os, sys

import theano

dirs = []
if len(sys.argv)>1:
    for compiledir in sys.argv[1:]:
        dirs.extend([os.path.join(compiledir,d) for d in os.listdir(compiledir)])
else:
    dirs = os.listdir(theano.config.compiledir)
    dirs = [os.path.join(theano.config.compiledir,d) for d in dirs]
keys = {} # key -> nb seen
mods = {}
for dir in dirs:

    try:
        f = open(os.path.join(dir, "key.pkl"))
        key = f.read()
        f.close()
        keys.setdefault(key, 0)
        keys[key]+=1
        del key
        del f
    except IOError:
        #print dir, "don't have a key.pkl file"
        pass
    try:
        path = os.path.join(dir, "mod.cpp")
        if not os.path.exists(path):
            path = os.path.join(dir, "mod.cu")
        f = open(path)
        mod = f.read()
        f.close()
        mods.setdefault(mod, 0)
        mods[mod]+=1
        del mod
        del f
        del path
    except IOError:
        print dir, "don't have a mod.{cpp,cu} file"
        pass


nbs = {} # nb seen -> now many key
for val in keys.values():
    nbs.setdefault(val, 0)
    nbs[val]+=1

print "key.pkl histograph"
print nbs

nbs = {} # nb seen -> now many key
more_then_one = 0
for val in mods.values():
    nbs.setdefault(val, 0)
    nbs[val]+=1
    if val>1:
        more_then_one += 1

print "mod.{cpp,cu} histogram"
print nbs
total = sum(mods.values())
uniq = len(mods)
useless = total - uniq
print "mod.{cpp,cu} total:", total
print "mod.{cpp,cu} uniq:", uniq
print "mod.{cpp,cu} with more then 1 copy:", more_then_one
print "mod.{cpp,cu} useless:", useless, float(useless)/total*100,"%"

print "nb directory", len(dirs)
