from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages, #Extension, Library
setup(name="Theano",
      version="0.1",
      description="Theano",
      long_description="""Machine learning toolkit""",
      author="LISA",
      author_email="lisa@iro.umontreal.ca",
      packages=find_packages(exclude='tests'),
      #scripts=['pygmy/audio/calc_feat.py'],

)

