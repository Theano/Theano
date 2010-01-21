import os
import ConfigParser

userconf_filename=""

default_={
'ProfileMode.n_apply_to_print':15,
'ProfileMode.n_ops_to_print':20,
'tensor_opt.local_elemwise_fusion':False,
'lib.amdlibm':False,
'op.set_flops':False,#currently used only in ConvOp. The profile mode will print the flops/s for the op.
'nvcc.fastmath':False,
'gpuelemwise.sync':True, #when true, wait that the gpu fct finished and check it error code.
}

#default value taked from env variable
THEANO_UNITTEST_SEED = os.getenv('THEANO_UNITTEST_SEED', 666)
THEANO_NOCLEANUP = os.getenv('THEANO_NOCLEANUP', 0)
THEANO_COMPILEDIR = os.getenv('THEANO_COMPILEDIR', None)
THEANO_BASE_COMPILEDIR = os.getenv('THEANO_BASE_COMPILEDIR', None)

HOME = os.getenv('HOME')

#0 compare with default precission, 1 less precission, 2 event less.
THEANO_CMP_SLOPPY = int(os.getenv('THEANO_CMP_SLOPPY', 0))

#flag for compiling with an optimized blas library. Used for gemm operation
#if THEANO_BLAS_LDFLAGS exist but empty, we will use numpy.dot()
THEANO_BLAS_LDFLAGS = os.getenv('THEANO_BLAS_LDFLAGS','-lblas')

#for gpu
CUDA_ROOT = os.getenv('CUDA_ROOT')
THEANO_GPU = os.getenv("THEANO_GPU")

THEANO_DEFAULT_MODE = os.getenv('THEANO_DEFAULT_MODE','FAST_RUN')

#debug mode
THEANO_DEBUGMODE_PATIENCE = int(os.getenv('THEANO_DEBUGMODE_PATIENCE', 10))
THEANO_DEBUGMODE_CHECK_C = bool(int(os.getenv('THEANO_DEBUGMODE_CHECK_C', 1)))
THEANO_DEBUGMODE_CHECK_PY = bool(int(os.getenv('THEANO_DEBUGMODE_CHECK_PY', 1)))
THEANO_DEBUGMODE_CHECK_FINITE = bool(int(os.getenv('THEANO_DEBUGMODE_CHECK_FINITE', 1)))
THEANO_DEBUGMODE_CHECK_STRIDES = bool(int(os.getenv('THEANO_DEBUGMODE_CHECK_STRIDES', 1)))

THEANO_FLAGS=os.getenv("THEANO_FLAGS","")

def parse_env_flags(flags, name , default_value=None):
    #The value in the env variable THEANO_FLAGS override the previous value
    val = default_value
    for flag in flags.split(','):
        if not flag:
            continue
        sp=flag.split('=',1)
        if sp[0]==name:
            if len(sp)==1:
                val=True
            else:
                val=sp[1]
            val=str(val)
    return val

floatX=parse_env_flags(THEANO_FLAGS,'floatX','float64')

class TheanoConfig(object):
    """Return the value for a key after parsing ~/.theano.cfg and 
    the THEANO_FLAGS environment variable.
    
    We parse in that order the value to have:
    1)the pair 'section.option':value in default_ 
    2)The ~/.theano.cfg file
    3)The value value provided in the get*() fct.
    The last value found is the value returned.
    
    The THEANO_FLAGS environement variable should be a list of comma-separated [section.]option[=value] entries. If the section part is omited, their should be only one section with that contain the gived option.
    """

    def __init__(self):
        d={} # no section
        for k,v in default_.items():
            if len(k.split('.'))==1:
                d[k]=v

        #set default value common for all section
        self.config = ConfigParser.SafeConfigParser(d)
        
        #set default value specific for each section
        for k, v in default_.items():
            sp = k.split('.',1)
            if len(sp)==2:
                if not self.config.has_section(sp[0]):
                    self.config.add_section(sp[0])
                self.config.set(sp[0], sp[1], str(v))


        #user config file override the default value
        self.config.read(['theano.cfg', os.path.expanduser('~/.theano.cfg')])

        self.env_flags=THEANO_FLAGS
        #The value in the env variable THEANO_FLAGS override the previous value
        for flag in self.env_flags.split(','):
            if not flag:
                continue
            sp=flag.split('=',1)
            if len(sp)==1:
                val=True
            else:
                val=sp[1]
            val=str(val)
            sp=sp[0].split('.',1)#option or section.option
            if len(sp)==2:
                self.config.set(sp[0],sp[1],val)
            else:
                found=0
                sp=sp[0].lower()#the ConfigParser seam to use only lower letter.
                for sec in self.config.sections():
                    for opt in self.config.options(sec):
                        if opt == sp:
                            found+=1
                            section=sec
                            option=opt
                if found==1:
                    self.config.set(section,option,val)
                elif found>1:
                    raise Exception("Ambiguous option (%s) in THEANO_FLAGS"%(sp))
                
    def __getitem__(self, key):
        """:returns: a str with the value associated to the key"""
        return self.get(key)

    def get(self, key, val=None):
        """ 
        :param key: the key that we want the value
        :type key: str

        :returns: a str with the value associated to the key
        """
        #self.config.get(section, option, raw, vars)
        if val is not None:
            return val
        sp = key.split('.',1)
        if len(sp)!=2:
            raise Exception("When we get a key, their must be a section and an option")
        return self.config.get(sp[0],sp[1], False)

    def getfloat(self, key, val=None):
        """ :return: cast the output of self.get to a float"""
        if val is not None:
            return float(val)
        return float(self.get(key))

    def getboolean(self, key, val=None):
        """ :return: cast the output of self.get to a boolean"""
        if val is None:
            val=self.get(key)
        if val == "False" or val == "0" or not val:
            val = False
        else:
            val = True
        return val

    def getint(self, key, val=None):
        """ :return: cast the output of self.get to an int"""
        if val is not None:
            return int(val)
        return int(self.get(key))

config = TheanoConfig()

if floatX not in ['float32', 'float64']:
    raise Exception("the configuration scalar.floatX must have value float32 or float64 not", floatX)

