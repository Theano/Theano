
import gof
import pprint
from gof import utils
from copy import copy
import re

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
        vars = dict(self.__vars__)
        for symbol, replacement in symequiv.iteritems():
            if replacement is not None:
                vars[replacement] = self.get(symbol)
        for symbol in set(symequiv.keys()).difference(set(symequiv.values())):
            del vars[symbol]
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
        return "\n".join(strings)

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

