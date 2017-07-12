from nose.plugins.skip import SkipTest
# NB: We raise a SkipTest (instead of another type of exception) because we're in a folder,
# thus nosetests will look for test files into this folder. With a SkipTest raised,
# the folder will be skipped by nosetests without failing.
raise SkipTest(
    "You are importing theano.sandbox.cuda. This is the old GPU back-end and "
    "is removed from Theano. Use Theano 0.9 to use it. Even better, "
    "transition to the new GPU back-end! See "
    "https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29")
