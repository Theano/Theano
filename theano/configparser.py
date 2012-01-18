# For flag of bool type, we consider the string 'False','false' and '0' as False
# and the string 'True', 'true', '1' as true.
# We also accept the bool type as its corresponding value!

import os, StringIO, sys
import ConfigParser
import logging
import warnings

import theano

_logger = logging.getLogger('theano.configparser')

class TheanoConfigWarning(Warning):

    def warn(cls, message, stacklevel=0):
        warnings.warn(message, cls, stacklevel=stacklevel + 3)
    warn = classmethod(warn)

# Check for deprecated environment variables
for key in os.environ:
    if key.startswith("THEANO"):
        if key not in ("THEANO_FLAGS", "THEANORC"):
            TheanoConfigWarning.warn("Ignoring deprecated environment variable %s" % key)

THEANO_FLAGS = os.getenv("THEANO_FLAGS", "")
# The THEANO_FLAGS environment variable should be a list of comma-separated
# [section.]option=value entries. If the section part is omitted, their should be only one
# section that contains the given option.

def parse_config_string(config_string, issue_warnings=True):
    """
    Parses a config string composed of comma-separated key=value components into a dict.
    """
    config_dict = {}
    for kv_pair in THEANO_FLAGS.split(','):
        kv_pair = kv_pair.strip()
        if not kv_pair:
            continue
        kv_tuple = kv_pair.split('=', 1)
        if len(kv_tuple) == 1:
            if issue_warnings:
                TheanoConfigWarning.warn("Config key '%s' has no value, ignoring it" % kv_tuple[0], stacklevel=1)
        else:
            k, v = kv_tuple
            # subsequent values for k will override earlier ones
            config_dict[k] = v
    return config_dict

THEANO_FLAGS_DICT = parse_config_string(THEANO_FLAGS, issue_warnings=True)

# THEANORC can contain a colon-delimited list of config files, like
# THEANORC=~lisa/.theanorc:~/.theanorc
# In that case, definitions in files on the right (here, ~/.theanorc) have
# precedence over those in files on the left.
def config_files_from_theanorc():
    rval = [os.path.expanduser(s) for s in os.getenv('THEANORC', '~/.theanorc').split(os.pathsep)]
    if os.getenv('THEANORC') is None and sys.platform == "win32":
        # to don't need to change the filename and make it open easily
        rval.append(os.path.expanduser('~/.theanorc.txt'))
    return rval


config_files = config_files_from_theanorc()
theano_cfg = ConfigParser.SafeConfigParser({'USER': os.getenv("USER", os.path.split(os.path.expanduser('~'))[-1])})
theano_cfg.read(config_files)
# Having a raw version of the config around as well enables us to pass
# through config values that contain format strings.
# The time required to parse the config twice is negligible.
theano_raw_cfg = ConfigParser.RawConfigParser()
theano_raw_cfg.read(config_files)


def fetch_val_for_key(key):
    """Return the overriding config value for a key.
    A successful search returns a string value.
    An unsuccessful search raises a KeyError

    The (decreasing) priority order is:
    - THEANO_FLAGS
    - ~./theanorc

    """

    # first try to find it in the FLAGS
    try:
        return THEANO_FLAGS_DICT[key]
    except KeyError:
        pass

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
        try:
            return theano_cfg.get(section, option)
        except ConfigParser.InterpolationError:
            return theano_raw_cfg.get(section, option)
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        raise KeyError(key)

_config_var_list = []

def _config_print(thing, buf):
    for cv in _config_var_list:
        print >> buf, cv
        print >> buf, "    Doc: ", cv.doc
        print >> buf, "    Value: ", cv.val
        print >> buf, ""


def get_config_md5():
    """
    Return a string md5 of the current config options. It should be such that
    we can safely assume that two different config setups will lead to two
    different strings.

    We only take into account config options for which `in_c_key` is True.
    """
    all_opts = sorted([c for c in _config_var_list if c.in_c_key],
                      key=lambda cv: cv.fullname)
    return theano.gof.cc.hash_from_code('\n'.join(
                    ['%s = %s' % (cv.fullname, cv.val) for cv in all_opts]))


class TheanoConfigParser(object):
    #properties are installed by AddConfigVar
    _i_am_a_config_class = True
    def __str__(self):
        sio = StringIO.StringIO()
        _config_print(self.__class__, sio)
        return sio.getvalue()

# N.B. all instances of TheanoConfigParser give access to the same properties.
config = TheanoConfigParser()

#
# The data structure at work here is a tree of CLASSES with CLASS ATTRIBUTES/PROPERTIES that
# are either a) INSTANTIATED dynamically-generated CLASSES, or b) ConfigParam instances.
# The root of this tree is the TheanoConfigParser CLASS, and the internal nodes are the SubObj
# classes created inside of AddConfigVar().
# Why this design ?
# - The config object is a true singleton.  Every instance of TheanoConfigParser is an empty
#   instance that looks up attributes/properties in the [single] TheanoConfigParser.__dict__
# - The subtrees provide the same interface as the root
# - ConfigParser subclasses control get/set of config properties to guard against craziness.

def AddConfigVar(name, doc, configparam, root=config, in_c_key=True):
    """Add a new variable to theano.config

    :type name: string for form "[section0.[section1.[etc]]].option"
    :param name: the full name for this configuration variable.

    :type doc: string
    :param doc: What does this variable specify?

    :type configparam: ConfigParam instance
    :param configparam: an object for getting and setting this configuration parameter

    :type root: object
    :param root: used for recusive calls -- do not provide an argument for this parameter.

    :type in_c_key: boolean
    :param in_c_key: If True, then whenever this config option changes, the
    key associated to compiled C modules also changes, i.e. it may trigger a
    compilation of these modules (this compilation will only be partial if it
    turns out that the generated C code is unchanged). Set this option to False
    only if you are confident this option should not affect C code compilation.

    :returns: None
    """

    # this method also performs some of the work of initializing ConfigParam instances

    if root is config:
        #only set the name in the first call, not the recursive ones
        configparam.fullname = name
    sections = name.split('.')
    if len(sections) > 1:
        # set up a subobject
        if not hasattr(root, sections[0]):
            # every internal node in the config tree is an instance of its own unique class
            class SubObj(object):
                _i_am_a_config_class = True
            setattr(root.__class__, sections[0], SubObj())
        newroot = getattr(root, sections[0])
        if not getattr(newroot, '_i_am_a_config_class', False) or isinstance(newroot, type):
            raise TypeError('Internal config nodes must be config class instances', newroot)
        return AddConfigVar('.'.join(sections[1:]), doc, configparam,
                            root=newroot, in_c_key=in_c_key)
    else:
        if hasattr(root, name):
            raise AttributeError('This name is already taken', configparam.fullname)
        configparam.doc = doc
        configparam.in_c_key = in_c_key
        configparam.__get__() # trigger a read of the value from config files and env vars
        setattr(root.__class__, sections[0], configparam)
        _config_var_list.append(configparam)


class ConfigParam(object):

    def __init__(self, default, filter=None,  allow_override=True):
        """
        If allow_override is False, we can't change the value after the import of Theano.
        So the value should be the same during all the execution
        """
        self.default = default
        self.filter = filter
        self.allow_override = allow_override
        # N.B. --
        # self.fullname  # set by AddConfigVar
        # self.doc       # set by AddConfigVar

        # Note that we do not call `self.filter` on the default value: this
        # will be done automatically in AddConfigVar, potentially with a
        # more appropriate user-provided default value.
        # Calling `filter` here may actually be harmful if the default value is
        # invalid and causes a crash or has unwanted side effects.

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
        if not self.allow_override and hasattr(self, 'val'):
            raise Exception("Can't change the value of this config parameter after initialization!")
        #print "SETTING PARAM", self.fullname,(cls), val
        if self.filter:
            self.val = self.filter(val)
        else:
            self.val = val


class EnumStr(ConfigParam):
    def __init__(self, default, *options, **kwargs):
        self.default = default
        self.all = (default,) + options

        # All options should be strings
        for val in self.all:
            if not isinstance(val, basestring):
                raise ValueError('Valid values for an EnumStr parameter '
                        'should be strings', val, type(val))

        def filter(val):
            if val in self.all:
                return val
            else:
                raise ValueError('Invalid value ("%s") for configuration variable "%s". Legal options are %s'
                        % (val, self.fullname, self.all))
        over = kwargs.get("allow_override", True)
        super(EnumStr, self).__init__(default, filter, over)

    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.all)

class TypedParam(ConfigParam):
    def __init__(self, default, mytype, is_valid=None, allow_override=True):
        self.mytype = mytype
        def filter(val):
            cast_val = mytype(val)
            if callable(is_valid):
                if is_valid(cast_val):
                    return cast_val
                else:
                    raise ValueError('Invalid value (%s) for configuration variable "%s".'
                            % (val, self.fullname), val)
            return cast_val
        super(TypedParam, self).__init__(default, filter,  allow_override=allow_override)
    def __str__(self):
        return '%s (%s) ' % (self.fullname, self.mytype)

def StrParam(default, is_valid=None, allow_override=True):
    return TypedParam(default, str, is_valid, allow_override=allow_override)
def IntParam(default, is_valid=None, allow_override=True):
    return TypedParam(default, int, is_valid, allow_override=allow_override)
def FloatParam(default, is_valid=None, allow_override=True):
    return TypedParam(default, float, is_valid, allow_override=allow_override)
def BoolParam(default, is_valid=None, allow_override=True):
#see comment at the beggining of this file.
    def booltype(s):
        if s in ['False','false','0', False]:
            return False
        elif s in ['True','true','1', True]:
            return True

    def is_valid_bool(s):
        if s in ['False', 'false', '0', 'True', 'true', '1', False, True]:
            return True
        else:
            return False
    if is_valid is None:
        is_valid = is_valid_bool
    return TypedParam(default, booltype, is_valid, allow_override=allow_override)
