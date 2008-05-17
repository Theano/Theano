
import gof
import pprint
from gof import utils
from copy import copy
import re

def _zip(*lists):
    if not lists:
        return ((), ())
    else:
        return zip(*lists)



# x = ivector()
# y = ivector()
# e = x + y

# f = Formula(x = x, y = y, e = e)

# y = x + x
# g = Formula(x=x,y=y)

# x2 = x + x
# g = Formula(x=x, x2=x2)




class Formula(utils.object2):
    
    def __init__(self, symtable_d = {}, **symtable_kwargs):
        vars = dict(symtable_d, **symtable_kwargs) 
        self.__dict__['__vars__'] = vars
        self.inputs = []
        self.outputs = []
        for symbol, var in vars.iteritems():
            if not isinstance(var, gof.Result):
                raise TypeError("All variables must be Result instances.", {symbol: var})
            var.name = symbol
            if var.owner is None:
                self.inputs.append(var)
            else:
                self.outputs.append(var)
        for unmentioned in set(gof.graph.inputs(self.outputs)).difference(self.inputs):
            if not isinstance(unmentioned, gof.Value):
                raise Exception("unmentioned input var: %s (please specify all the variables involved in the formula except constants)" % unmentioned)


    #################
    ### VARIABLES ###
    #################

    input_names = property(lambda self: [r.name for r in self.inputs])
    output_names = property(lambda self: [r.name for r in self.outputs])

    def has_var(self, symbol):
        return symbol in self.__dict__['__vars__']

    def get(self, symbol):
        try:
            return self.__dict__['__vars__'][symbol]
        except KeyError:
            raise AttributeError("unknown var: %s" % symbol)

    def get_all(self, regex):
        regex = re.compile("^%s$" % regex)
        return [r for r in self.__vars__.values() if regex.match(r.name)]

    def __getattr__(self, attr):
        if self.has_var(attr):
            return self.get(attr)
        try:
            return self.__dict__[attr]
        except KeyError:
            raise AttributeError("no such attribute: %s" % attr)

    def __setattr__(self, attr, value):
        if self.has_var(attr):
            raise AttributeError("a Formula's variables are read-only (tried to set '%s')" % attr)
        self.__dict__[attr] = value


    ################
    ### RENAMING ###
    ################

    def __rename__(self, **symequiv):
#         print "~~~~~~~~~~~~~"
#         print symequiv
        vars = dict(self.__vars__)
        for symbol, replacement in symequiv.iteritems():
            if replacement is not None:
                vars[replacement] = self.get(symbol)
#         print vars
#         print set(symequiv.keys()).difference(set(symequiv.values()))
#         print set(symequiv.keys()), set(symequiv.values())
        for symbol in set(symequiv.keys()).difference(set(symequiv.values())):
            del vars[symbol]
#         print vars
        return Formula(vars)

    def rename(self, **symequiv):
        new_inputs = map(copy, self.inputs)
        return self.assign(**dict(zip([r.name for r in new_inputs], new_inputs))).__rename__(**symequiv)

    def rename_regex(self, rules, reuse_results = False):
        symequiv = {}
        for name in self.__vars__.keys():
            for rule, repl in rules.items():
                rule = "^%s$" % rule
                match = re.match(rule, name)
                if match:
                    if callable(repl):
                        symequiv[name] = repl(*match.groups())
                    else:
                        symequiv[name] = re.sub(rule, repl, name)
        if reuse_results:
            return self.__rename__(**symequiv)
        else:
            return self.rename(**symequiv)

    def normalize(self, reuse_results = False):
        return self.increment(0, reuse_results)

    def increment(self, n, reuse_results = False):
        def convert(name, prev_n):
            if prev_n == "": return "%s%i" % (name, 1 + n)
            else: return "%s%i" % (name, int(prev_n) + n)
        return self.rename_regex({"(.*?)([0-9]*)": convert}, reuse_results)

    def prefix(self, prefix, reuse_results = False):
        return self.rename_regex({"(.*)": ("%s\\1" % prefix)}, reuse_results)

    def suffix(self, suffix, push_numbers = False, reuse_results = False):
        if push_numbers:
            return self.rename_regex({"(.*?)([0-9]*)": ("\\1%s\\2" % suffix)}, reuse_results)
        else:
            return self.rename_regex({"(.*)": ("\\1%s" % suffix)}, reuse_results)


    ##################
    ### ASSIGNMENT ###
    ##################

    def assign(self, **assignment):
        try:
            inputs = [assignment.pop(name) for name in self.input_names]
        except KeyError, e:
            raise KeyError("missing input: '%s'" % name)
        if assignment:
            raise KeyError("unknown inputs: %s" % assignment)
        inputs, outputs = gof.graph.clone_with_equiv(self.inputs, self.outputs, dict(zip(self.inputs, inputs)))
        #inputs, outputs = gof.graph.clone_with_new_inputs(self.inputs, self.outputs, inputs)
        return Formula(zip(self.input_names, inputs) + zip(self.output_names, outputs))

    def reassign(self, **assignment):
        d = dict(zip(self.input_names, self.inputs))
        d.update(assignment)
        return self.assign(**d)

    def clone(self):
        return self.assign(**dict(zip(self.input_names, map(copy, self.inputs))))

    def glue(self, *formulas):
        return glue(self, *formulas)

    def __str__(self):
        strings = ["inputs: " + ", ".join(self.input_names)]
        for node in gof.graph.io_toposort(self.inputs, self.outputs):
            for output in node.outputs:
                if output in self.outputs:
                    strings.append("%s = %s" % (output,
                                                pprint.pp.clone_assign(lambda pstate, r: r.name in self.__vars__ and r is not output,
                                                                       pprint.LeafPrinter()).process(output)))
#                    strings.append("%s = %s" % (output,
#                                                pprint.pp.process(output)))
        #strings.append(str(gof.graph.as_string(self.inputs, self.outputs)))
        return "\n".join(strings)
#    (self.inputs + utils.difference(self.outputs, node.outputs),[output])[0]

    #################
    ### OPERATORS ###
    #################

    def __add__(self, other):
        if isinstance(other, int):
            return self.increment(other)
        elif isinstance(other, str):
            return self.suffix(other)
        return self.glue(other)

    def __radd__(self, other):
        if isinstance(other, int):
            return self.increment(other)
        elif isinstance(other, str):
            return self.prefix(other)
        return self.glue(other)

    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        return glue(*map(self.increment, range(0, n)))



    
def glue2(f1, f2):
    f1_new = dict(zip(f1.inputs, f1.inputs))
    f2_new = dict(zip(f2.inputs, f2.inputs))
    
    for r1 in f1.inputs:
        name = r1.name
        if f2.has_var(name):
            r2 = f2.get(name)
            #if not r1.type == r2.type:
            #    raise TypeError("inconsistent typing for %s: %s, %s" % (name, r1.type, r2.type))
            if name in f2.input_names:
                f2_new[r2] = r1
            elif name in f2.output_names:
                f1_new[r1] = r2

    for r1 in f1.outputs:
        name = r1.name
        if f2.has_var(name):
            r2 = f2.get(name)
            #if not r1.type == r2.type:
            #    raise TypeError("inconsistent typing for %s: %s, %s" % (name, r1.type, r2.type))
            if name in f2.input_names:
                f2_new[r2] = r1
            elif name in f2.output_names:
                raise Exception("It is not allowed for a variable to be the output of two different formulas: %s" % name)

    g = gof.graph
    i1, o1 = g.clone_with_equiv(f1.inputs, f1.outputs, f1_new)
    i2, o2 = g.clone_with_equiv(f2.inputs, f2.outputs, f2_new) #list(set(f2.inputs + g.inputs(o1)))
    
    vars = {}
    vars.update(zip(f1.input_names, i1))
    vars.update(zip(f1.output_names, o1))
    vars.update(zip(f2.input_names, i2))
    vars.update(zip(f2.output_names, o2))    

    try:
        return Formula(vars)
    except Exception, e:
        e.args = e.args + ("Circular dependencies might have caused this error.", )
        raise


def glue(*formulas):
    return reduce(glue2, formulas)


import tensor as T
sep = "---------------------------"



class FormulasMetaclass(type):
    def __init__(cls, name, bases, dct):
        variables = {}
        for name, var in dct.items():
            if isinstance(var, gof.Result):
                variables[name] = var
        cls.__variables__ = variables
        cls.__canon__ = Formula(cls.__variables__)

class Formulas(utils.object2):
    __metaclass__ = FormulasMetaclass
    def __new__(cls):
        return cls.__canon__.clone()





# class Test(Formulas):
#     x = T.ivector()
#     y = T.ivector()
#     e = x + y + 21

# x = T.ivector()
# y = T.ivector()
# e = x + y + 21

# f1 = Formula(x = x, y = y, e = e)

# Test() -> f1.clone()



# f = Test()
# print f
# print f.reassign(x = T.ivector())
# print f.reassign(x = T.dvector(), y = T.dvector())
# print f.reassign(x = T.dmatrix(), y = T.dmatrix())

# class Test(Formulas):
#     x = T.ivector()
#     e = x + 999

# f = Test()
# print f
# print f.reassign(x = T.ivector())
# print f.reassign(x = T.dvector())


# class Layer(Formulas):
#     x = T.ivector()
#     y = T.ivector()
#     x2 = x + y

# # print Layer()
# # print Layer() + 1
# # print Layer() + 2
# # print Layer() + 3

# print Layer() * 3


# def sigmoid(x):
#     return 1.0 / (1.0 + T.exp(-x))

#class Update(Formulas):
#    param = T.matrix()
#    lr, cost = T.scalars(2)
#    param_update = param - lr * T.sgrad(cost, param)

#class SumSqrDiff(Formulas):
#     target, output = T.rows(2)
#     cost = T.sum((target - output)**2)

# class Layer(Formulas):
#     input, bias = T.rows(2)
#     weights = T.matrix()
#     input2 = T.tanh(bias + T.dot(input, weights))

# forward = Layer()*2
# g = glue(forward.rename(input3 = 'output'),
#          SumSqrDiff().rename(target = 'input1'),
#          *[Update().rename_regex({'param(.*)': ('%s\\1' % param.name)}) for param in forward.get_all('(weight|bias).*')])
# sg = g.__str__()
# print unicode(g)





# lr = 0.01
# def autoassociator_f(x, w, b, c):
#     reconstruction = sigmoid(T.dot(sigmoid(T.dot(x, w) + b), w.T) + c)
#     rec_error = T.sum((x - reconstruction)**2)
#     new_w = w - lr * Th.grad(rec_error, w)
#     new_b = b - lr * Th.grad(rec_error, b)
#     new_c = c - lr * Th.grad(rec_error, c)
# #    f = Th.Function([x, w, b, c], [reconstruction, rec_error, new_w, new_b, new_c])
#     f = Th.Function([x, w, b, c], [reconstruction, rec_error, new_w, new_b, new_c], linker_cls = Th.gof.OpWiseCLinker)
#     return f

# x, w = T.matrices('xw')
# b, c = T.rows('bc')
# f = autoassociator_f(x, w, b, c)

# w_val, b_val, c_val = numpy.random.rand(10, 10), numpy.random.rand(1, 10), numpy.random.rand(1, 10)
# x_storage = numpy.ndarray((1, 10))

# for i in dataset_1hot(x_storage, numpy.ndarray((1, )), 10000):
#     rec, err, w_val, b_val, c_val = f(x_storage, w_val, b_val, c_val)
#     if not(i % 100):
#         print err




























# x = T.ivector()
# y = T.ivector()
# z = x + y
# e = z - 24

# f = Formula(x = x, y = y, z = z, e = e)
# print f

# print sep

# a = T.lvector()
# b = a * a
# f2 = Formula(e = a, b = b)
# print f2

# print sep

# print glue(f, f2)

# print sep



# x1 = T.ivector()
# y1 = x1 + x1

# y2 = T.ivector()
# x2 = y2 + y2

# f1 = Formula(x=x1, y=y1)
# f2 = Formula(x=x2, y=y2)

# print f1
# print sep
# print f2
# print sep
# print glue(f1, f2)
# print sep



# x1 = T.ivector()
# z1 = T.ivector()
# y1 = x1 + z1
# w1 = x1 * x1

# x2 = T.ivector()
# e2 = T.ivector()
# z2 = e2 ** e2
# g2 = z2 + x2

# f1 = Formula(x=x1, z=z1, y=y1, w=w1)
# f2 = Formula(x=x2, e=e2, z=z2, g=g2)

# print sep
# print f1
# print sep
# print f2
# print sep
# print f1 + f2



# x1 = T.ivector()
# z1 = T.ivector()
# y1 = x1 + x1
# w1 = z1 + z1

# e2 = T.ivector()
# w2 = T.ivector()
# x2 = e2 + e2
# g2 = w2 + w2

# f1 = Formula(x=x1, z=z1, y=y1, w=w1)
# f2 = Formula(x=x2, e=e2, w=w2, g=g2)

# print sep
# print f1
# print sep
# print f2
# print sep
# print f1 + f2








# def glue2(f1, f2):
#     reassign_f1 = {}
#     reassign_f2 = {}
#     equiv = {}
#     for r1 in f1.inputs:
#         name = r1.name
#         try:
#             r2 = f2.get(name)
#             if not r1.type == r2.type:
#                 raise TypeError("inconsistent typing for %s: %s, %s" % (name, r1.type, r2.type))
#             if name in f2.input_names:
#                 reassign_f2[r2] = r1
#             elif name in f2.output_names:
#                 reassign_f1[r1] = r2
#         except AttributeError:
#             pass
#     for r1 in f1.outputs:
#         name = r1.name
#         try:
#             r2 = f2.get(name)
#             if not r1.type == r2.type:
#                 raise TypeError("inconsistent typing for %s: %s, %s" % (name, r1.type, r2.type))
#             if name in f2.input_names:
#                 reassign_f2[r2] = r1
#             elif name in f2.output_names:
#                 raise Exception("It is not allowed for a variable to be the output of two different formulas: %s" % name)
#         except AttributeError:
#             pass
#     print reassign_f1
#     print reassign_f2
#     #i0, o0 = gof.graph.clone_with_new_inputs(f1.inputrs+f2.inputrs, f1.outputrs+f2.outputrs,
#     #                                         [reassign_f1.get(name, r) for name, r in zip(f1.inputs, f1.inputrs)] +
#     #                                         [reassign_f2.get(name, r) for name, r in zip(f2.inputs, f2.inputrs)])
#     #print gof.Env([x for x in i0 if x.owner is None], o0)
#     #return
#     ##equiv = gof.graph.clone_with_new_inputs_get_equiv(f1.inputrs, f1.outputrs, [reassign_f1.get(name, r) for name, r in zip(f1.inputs, f1.inputrs)])
#     ##i1, o1 = [equiv[r] for r in f1.inputrs], [equiv[r] for r in f1.outputrs]
#     ##_reassign_f2, reassign_f2 = reassign_f2, {}
#     ##for name, r in _reassign_f2.items():
#     ##    print name, r, equiv.get(r, r) is r
#     ##    reassign_f2[name] = equiv.get(r, r)

# ##    i1, o1 = gof.graph.clone_with_new_inputs(f1.inputs, f1.outputs, [reassign_f1.get(name, r) for name, r in zip(f1.input_names, f1.inputs)])
# ##    i2, o2 = gof.graph.clone_with_new_inputs(f2.inputs, f2.outputs, [reassign_f2.get(name, r) for name, r in zip(f2.input_names, f2.inputs)])
#     i1, o1 = gof.graph.clone_with_equiv(f1.inputs, f1.outputs, reassign_f1)
#     vars = {}
#     vars.update(zip(f1.input_names, i1))
#     vars.update(zip(f1.output_names, o1))
#     vars.update(zip(f2.input_names, i2))
#     vars.update(zip(f2.output_names, o2))
#     #print vars
#     #print gof.graph.as_string(i1, o1)
#     #print gof.graph.as_string(i2, o2)
#     #print "a"
#     #o = o1[0]
#     #while o.owner is not None:
#     #    print o, o.owner
#     #    o = o.owner.inputs[0]
#     #print "b", o
#     return Formula(vars)
    







# class FormulasMetaclass(type):
#     def __init__(cls, name, bases, dct):
#         variables = {}
#         for name, var in dct.items():
#             if isinstance(var, gof.Result):
#                 variables[name] = var
#         cls.__variables__ = variables
#         cls.__canon__ = Formula(cls.__variables__)

# class Formulas(utils.object2):
#     __metaclass__ = FormulasMetaclass
#     def __new__(cls):
#         return cls.__canon__.clone()



# class Test(Formulas):
#     x = T.ivector()
#     y = T.ivector()
#     e = x + y

# class Test2(Formulas):
#     e = T.ivector()
#     x = T.ivector()
#     w = e ** (x / e)

# f = Test() # + Test2()

# print f
# print sep

# print f.prefix("hey_")
# print sep

# print f.suffix("_woot")
# print sep

# print f.increment(1)
# print sep

# print f.normalize()
# print sep

# print f.increment(1).increment(1)
# print sep

# print f.increment(8).suffix("_yep")
# print sep

# print (f + 8).suffix("_yep", push_numbers = True)
# print sep

# print f.suffix("_yep", push_numbers = True)
# print sep

# print f.rename_regex({"(x|y)": "\\1\\1",
#                       'e': "OUTPUT"})
# print sep

# print f + "_suffix"
# print sep

# print "prefix_" + f
# print sep



#### Usage case ####

# class Forward1(Formula):
#     input, b, c = drows(3)
#     w = dmatrix()
#     output = dot(sigmoid(dot(w, input) + b), w.T) + c

# class Forward2(Formula):
#     input, b, c = drows(3)
#     w1, w2 = dmatrices(2)
#     output = dot(sigmoid(dot(w1, input) + b), w2) + c

# class SumSqrError(Formula):
#     target, output = drows(2)
#     cost = sum((target - output)**2)

# class GradUpdate(Formula):
#     lr, cost = dscalars(2)
#     param = dmatrix()
#     param_updated = param + lr * grad(cost, param)


# NNetUpdate = glue(Forward1(), SumSqrError(), [GradUpdate.rename(param = name) for name in ['w', 'b', 'c']])








# class Forward(Formula):
#     input, w, b, c = vars(4)
#     output = dot(sigmoid(dot(w, input) + b), w.T) + c

# class SumSqrError(Formula):
#     target, output = vars(2)
#     cost = sum((target - output)**2)

# class GradUpdate(Formula):
#     lr, cost, param = vars(3)
#     param_updated = param + lr * grad(cost, param)

# NNetUpdate = Forward() + SumSqrError() + [GradUpdate().rename({'param*': name}) for name in 'wbc']


# #NNetUpdate = Forward() + SumSqrError() + [GradUpdate().rename(param = name) for name in ['w', 'b', 'c']]

