import os, StringIO, sys
import ConfigParser
import logging
_logger = logging.getLogger('theano.config')

for key in os.environ:
    if key.startswith("THEANO"):
        if key not in ("THEANO_FLAGS", "THEANORC"):
            print >> sys.stderr, "ERROR: Ignoring deprecated environment variable", key


THEANO_FLAGS=os.getenv("THEANO_FLAGS","")
# The THEANO_FLAGS environement variable should be a list of comma-separated
# [section.]option[=value] entries. If the section part is omited, their should be only one
# section with that contain the gived option.

theano_cfg_path = os.getenv('THEANORC', '~/.theanorc')
theano_cfg = ConfigParser.SafeConfigParser()
theano_cfg.read([os.path.expanduser(theano_cfg_path)])

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

def fetch_val_for_key(key):
    """Return the overriding config value for a key.
    A successful search returs a string value.
    An unsuccessful search raises a KeyError
    
    The priority order is:
    - THEANO_FLAGS
    - ~./theanorc
    
    """

    # first try to find it in the FLAGS
    for name_val in THEANO_FLAGS.split(','):
        if not name_val:
            continue
        name_val_tuple=name_val.split('=',1)
        if len(name_val_tuple)==1:
            name, val = name_val_tuple, str(True)
        else:
            name, val = name_val_tuple

        if name == key:
            return val

    # next try to find it in the config file

    # config file keys can be of form option, or section.option
    key_tokens = key.split('.')
    if len(key_tokens) > 2:
        raise KeyError(key)

    if len(key_tokens) == 2:
        section, option = key_tokens
    else:
        section, option = 'global', key
    try:
        return theano_cfg.get(section, option)
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        raise KeyError(key)
    
class TheanoConfigParser(object):
    #properties are installed by AddConfigVar

    def __str__(self):
        sio = StringIO.StringIO()
        _config_print(self.__class__, sio)
        return sio.getvalue()
    pass
# N.B. all instances of TheanoConfigParser give access to the same properties.
config = TheanoConfigParser()

_config_var_list = []

def _config_print(thing, buf):
    for cv in _config_var_list:
        print >> buf, cv
        print >> buf, "    Doc: ", cv.doc
        print >> buf, "    Value: ", cv.val
        print >> buf, ""


def AddConfigVar(name, doc, thing, cls=TheanoConfigParser):
    if cls == TheanoConfigParser:
        thing.fullname = name
    if hasattr(TheanoConfigParser, name):
        raise ValueError('This name is already taken')
    parts = name.split('.')
    if len(parts) > 1:
        # set up a subobject
        if not hasattr(cls, parts[0]):
            class SubObj(object):
                pass
            setattr(cls, parts[0], SubObj)
        AddConfigVar('.'.join(parts[1:]), doc, thing, cls=getattr(cls, parts[0]))
    else:
        thing.doc = doc
        thing.__get__() # trigger a read of the value
        setattr(cls, parts[0], thing)
        _config_var_list.append(thing)

class ConfigParam(object):
    def __init__(self, default, filter=None):
        self.default = default
        self.filter=filter
        # there is a name attribute too, but it is set by AddConfigVar

    def __get__(self, *args):
        #print "GETTING PARAM", self.fullname, self, args
        if not hasattr(self, 'val'):
            try:
                val_str = fetch_val_for_key(self.fullname)
            except KeyError:
                val_str = self.default
            self.__set__(None, val_str)
        #print "RVAL", self.val
        return self.val

    def __set__(self, cls, val):
        #print "SETTING PARAM", self.fullname,(cls), val
        if self.filter:
            self.val = self.filter(val)
        else:
            self.val = val

    deleter=None
    
class EnumStr(ConfigParam):
    def __init__(self, default, *options):
        self.default = default
        self.all = (default,) + options
        def filter(val):
            if val in self.all:
                return val
            else:
                raise ValueError('Invalid value (%s) for configuration variable "%s". Legal options are %s'
                        % (val, self.fullname, self.all), val)
        super(EnumStr, self).__init__(default, filter)

    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.all)

class TypedParam(ConfigParam):
    def __init__(self, default, mytype, is_valid=None):
        self.mytype = mytype
        def filter(val):
            casted_val = mytype(val)
            if callable(is_valid):
                if is_valid(casted_val):
                    return casted_val
                else:
                    raise ValueError('Invalid value (%s) for configuration variable "%s".'
                            % (val, self.fullname), val)
            return casted_val
        super(TypedParam, self).__init__(default, filter)
    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.mytype)

def StrParam(default, is_valid=None):
    return TypedParam(default, str, is_valid)
def IntParam(default, is_valid=None):
    return TypedParam(default, int, is_valid)
def FloatParam(default, is_valid=None):
    return TypedParam(default, float, is_valid)
def BoolParam(default, is_valid=None):
    return TypedParam(default, bool, is_valid)
