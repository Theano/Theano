from nose.plugins.skip import SkipTest

raise SkipTest(
    "You are importing theano.sandbox.cuda. This is the old GPU back-end and "
    "is removed from Theano. Use Theano 0.9 to use it. Even better, "
    "transition to the new GPU back-end! See "
    "https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29")
