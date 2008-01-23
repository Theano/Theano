
from scipy.weave import c_spec, standard_array_spec


class omega_type_converter_extension:

    def provides(self):
        """
        Returns a list of (c_type, name, init_code) tuples that represent variables
        the type converter provides to the user's code.
        """
        tvars = self.template_vars()
        return [(tvars['c_type'], tvars['name'], tvars['var_convert'])]

    def format_provide(self, x):
        return '%s %s = %s;' % x

    def declaration_code(self, templatize = 0, inline = 0):
        tvars = self.template_vars(inline=inline)
        code = '%(py_var)s = %(var_lookup)s;\n' % tvars
        code += '\n'.join([self.format_provide(export) for export in self.provides()])
        return code

    def struct_members_code(self):
        return '\n'.join(['%s_type %s;' % (name, name) for c_type, name, init in self.provides()])

    def struct_import_code(self):
        return '\n'.join(['__STRUCT_P->%s = %s;' % (name, name) for c_type, name, init in self.provides()])

    def struct_support_code(self):
        return ""

    def struct_declaration_code(self):
        return ""

    def struct_typedefs(self):
        return "\n".join(["typedef %s %s_type;" % (c_type, name) for c_type, name, init in self.provides()])

#     def struct_template_types(self):
#         return [("typename %s_type" % name, ) for c_type, name, init in self.provides()]


class int_converter(omega_type_converter_extension, c_spec.int_converter):
    pass

class float_converter(omega_type_converter_extension, c_spec.float_converter):
    pass

class complex_converter(omega_type_converter_extension, c_spec.complex_converter):
    pass

class unicode_converter(omega_type_converter_extension, c_spec.unicode_converter):
    def provides(self):
        tvars = self.template_vars()
        return omega_type_converter_extension.provides() + [('int', 'N%(name)s' % tvars, 'PyUnicode_GET_SIZE(%(py_var)s)' % tvars)]

class string_converter(omega_type_converter_extension, c_spec.string_converter):
    pass

class list_converter(omega_type_converter_extension, c_spec.list_converter):
    pass

class dict_converter(omega_type_converter_extension, c_spec.dict_converter):
    pass

class tuple_converter(omega_type_converter_extension, c_spec.tuple_converter):
    pass

class file_converter(omega_type_converter_extension, c_spec.file_converter):
    pass

class instance_converter(omega_type_converter_extension, c_spec.instance_converter):
    pass

class array_converter(omega_type_converter_extension, standard_array_spec.array_converter):
    def provides(self):
        tvars = self.template_vars()
        ret = []
        ret.append((tvars['c_type'], tvars['array_name'], tvars['var_convert']))
        ret.append(('npy_intp*', 'N%(name)s' % tvars, '%(array_name)s->dimensions' % tvars))
        ret.append(('npy_intp*', 'S%(name)s' % tvars, '%(array_name)s->strides' % tvars))
        ret.append(('int', 'D%(name)s' % tvars, '%(array_name)s->nd' % tvars))
        ret.append(('%(num_type)s*' % tvars, '%(name)s' % tvars, '(%(num_type)s*) %(array_name)s->data' % tvars))
        return ret

    def declaration_code(self, templatize = 0, inline = 0):
        tvars = self.template_vars(inline=inline)
        tvars['cap_name'] = self.name.upper()
        prov = self.provides()
        code = '%(py_var)s = %(var_lookup)s;\n' % tvars
        code += "\n".join(self.format_provide(export) for export in prov[:1])
        code += '\nconversion_numpy_check_type(%(array_name)s,%(num_typecode)s,"%(name)s");\n' % tvars
        code += "\n".join(self.format_provide(export) for export in prov[1:])
        return code
        
    def struct_support_code(self, templatize = 0, inline = 0):
        tvars = self.template_vars(inline=inline)
        cap_name = self.name.upper()
        tvars['cap_name'] = cap_name        
        code = 'inline %(num_type)s& %(cap_name)s1(int i) { return (*((%(num_type)s*)(%(array_name)s->data + (i)*S%(name)s[0])));}\n' \
               'inline %(num_type)s& %(cap_name)s2(int i, int j) { return (*((%(num_type)s*)(%(array_name)s->data + (i)*S%(name)s[0] + (j)*S%(name)s[1])));}\n' \
               'inline %(num_type)s& %(cap_name)s3(int i, int j, int k) { return (*((%(num_type)s*)(%(array_name)s->data + (i)*S%(name)s[0] + (j)*S%(name)s[1] + (k)*S%(name)s[2])));}\n' \
               'inline %(num_type)s& %(cap_name)s4(int i, int j, int k, int l) { return (*((%(num_type)s*)(%(array_name)s->data + (i)*S%(name)s[0] + (j)*S%(name)s[1] + (k)*S%(name)s[2] + (l)*S%(name)s[3])));}\n'
        return code % tvars

    def struct_typedefs(self):
        tvars = self.template_vars()
        return omega_type_converter_extension.struct_typedefs(self) + "\n" + "typedef %(num_type)s %(name)s_dtype;" % tvars
    
#        return "\n".join(["typedef %s %s_type;" % (c_type, name)])
#     def struct_template_types(self):
#         tvars = self.template_vars()
#         return [("typename %s_type" % name, c_type) for c_type, name, init in self.provides()] + [("typename %s_dtype" % self.name, tvars['num_type'])]



default = [array_converter(),
           int_converter(),
           float_converter(),
           complex_converter(),
           unicode_converter(),
           string_converter(),
           list_converter(),
           dict_converter(),
           tuple_converter(),
           file_converter(),
           instance_converter()]

