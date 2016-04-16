from __future__ import absolute_import, print_function, division
from theano.configdefaults import short_platform


def test_short_platform():
    for r, p, a in [  # (release, platform, answer)
        ('3.2.0-70-generic',
         'Linux-3.2.0-70-generic-x86_64-with-debian-wheezy-sid',
         "Linux-3.2--generic-x86_64-with-debian-wheezy-sid"),
        ('3.2.0-70.1-generic',
         'Linux-3.2.0-70.1-generic-x86_64-with-debian-wheezy-sid',
         "Linux-3.2--generic-x86_64-with-debian-wheezy-sid"),
        ('3.2.0-70.1.2-generic',
         'Linux-3.2.0-70.1.2-generic-x86_64-with-debian-wheezy-sid',
         "Linux-3.2--generic-x86_64-with-debian-wheezy-sid"),
        ('2.6.35.14-106.fc14.x86_64',
         'Linux-2.6.35.14-106.fc14.x86_64-x86_64-with-fedora-14-Laughlin',
         'Linux-2.6-fc14.x86_64-x86_64-with-fedora-14-Laughlin'),
    ]:
        o = short_platform(r, p)
        assert o == a, (o, a)
