from __future__ import absolute_import, print_function, division
from theano.tests.record import Record, MismatchError, RecordMode
from theano import function
from six.moves import xrange, StringIO
from theano.tensor import iscalar


def test_record_good():
    # Tests that when we record a sequence of events, then
    # repeat it exactly, the Record class:
    #     1) Records it correctly
    #     2) Does not raise any errors

    # Record a sequence of events
    output = StringIO()

    recorder = Record(file_object=output, replay=False)

    num_lines = 10

    for i in xrange(num_lines):
        recorder.handle_line(str(i) + '\n')

    # Make sure they were recorded correctly
    output_value = output.getvalue()

    assert output_value == ''.join(str(i) + '\n' for i in xrange(num_lines))

    # Make sure that the playback functionality doesn't raise any errors
    # when we repeat them
    output = StringIO(output_value)

    playback_checker = Record(file_object=output, replay=True)

    for i in xrange(num_lines):
        playback_checker.handle_line(str(i) + '\n')


def test_record_bad():
    # Tests that when we record a sequence of events, then
    # do something different on playback, the Record class catches it.

    # Record a sequence of events
    output = StringIO()

    recorder = Record(file_object=output, replay=False)

    num_lines = 10

    for i in xrange(num_lines):
        recorder.handle_line(str(i) + '\n')

    # Make sure that the playback functionality doesn't raise any errors
    # when we repeat some of them
    output_value = output.getvalue()
    output = StringIO(output_value)

    playback_checker = Record(file_object=output, replay=True)

    for i in xrange(num_lines // 2):
        playback_checker.handle_line(str(i) + '\n')

    # Make sure it raises an error when we deviate from the recorded sequence
    try:
        playback_checker.handle_line('0\n')
    except MismatchError:
        return
    raise AssertionError("Failed to detect mismatch between recorded sequence "
                         " and repetition of it.")


def test_record_mode_good():
    # Like test_record_good, but some events are recorded by the
    # theano RecordMode. We don't attempt to check the
    # exact string value of the record in this case.

    # Record a sequence of events
    output = StringIO()

    recorder = Record(file_object=output, replay=False)

    record_mode = RecordMode(recorder)

    i = iscalar()
    f = function([i], i, mode=record_mode, name='f')

    num_lines = 10

    for i in xrange(num_lines):
        recorder.handle_line(str(i) + '\n')
        f(i)

    # Make sure that the playback functionality doesn't raise any errors
    # when we repeat them
    output_value = output.getvalue()
    output = StringIO(output_value)

    playback_checker = Record(file_object=output, replay=True)

    playback_mode = RecordMode(playback_checker)

    i = iscalar()
    f = function([i], i, mode=playback_mode, name='f')

    for i in xrange(num_lines):
        playback_checker.handle_line(str(i) + '\n')
        f(i)


def test_record_mode_bad():
    # Like test_record_bad, but some events are recorded by the
    # theano RecordMode, as is the event that triggers the mismatch
    # error.

    # Record a sequence of events
    output = StringIO()

    recorder = Record(file_object=output, replay=False)

    record_mode = RecordMode(recorder)

    i = iscalar()
    f = function([i], i, mode=record_mode, name='f')

    num_lines = 10

    for i in xrange(num_lines):
        recorder.handle_line(str(i) + '\n')
        f(i)

    # Make sure that the playback functionality doesn't raise any errors
    # when we repeat them
    output_value = output.getvalue()
    output = StringIO(output_value)

    playback_checker = Record(file_object=output, replay=True)

    playback_mode = RecordMode(playback_checker)

    i = iscalar()
    f = function([i], i, mode=playback_mode, name='f')

    for i in xrange(num_lines // 2):
        playback_checker.handle_line(str(i) + '\n')
        f(i)

    # Make sure a wrong event causes a MismatchError
    try:
        f(0)
    except MismatchError:
        return
    raise AssertionError("Failed to detect a mismatch.")
