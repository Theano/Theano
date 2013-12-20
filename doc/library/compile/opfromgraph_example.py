import theano
from theano import tensor, OpFromGraph


x, y, z = tensor.vectors("xyz")

out1 = x + y
out2 = x + z

new_op = OpFromGraph([x, y, z], [out1, out2])
# This apply new_op to x, y and z. outs is a list of output variable.
outs = new_op(x, y, z)
f1 = theano.function([x, y, z], [out1, out2])
f2 = theano.function([x, y, z], outs)

print "Normal graph"
theano.printing.debugprint(f1)
print "Encapsulated graph"
theano.printing.debugprint(f2)
