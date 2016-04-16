from __future__ import absolute_import, print_function, division

letters = [
    ('b', 'int8'),
    ('w', 'int16'),
    ('i', 'int32'),
    ('l', 'int64'),
    ('d', 'float64'),
    ('f', 'float32'),
    ('c', 'complex64'),
    ('z', 'complex128') ]

shapes = [
        ('scalar', ()),
        ('vector', (False,)),
        ('row', (True, False)),
        ('col', (False, True)),
        ('matrix', (False,False)),
        ('tensor3', (False,False,False)),
        ('tensor4', (False,False,False,False)),]

hdr = '============ =========== ==== =========== ================================='
print(hdr)
print('Constructor  dtype       ndim shape       broadcastable')
print(hdr)
for letter in letters:
    for shape in shapes:
        suff = ',)' if len(shape[1])==1 else ')'
        s = '(' + ','.join('1' if b else '?' for b in shape[1]) + suff
        print('%s%-10s  %-10s  %-4s %-10s  %-20s' %(
                letter[0], shape[0], letter[1], len(shape[1]), s, shape[1]
                ))
print(hdr)
