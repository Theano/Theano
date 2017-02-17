#!/usr/bin/env python
from __future__ import absolute_import, print_function, division

__docformat__ = 'restructuredtext en'

import difflib
import operator
import os
import string
from subprocess import Popen, PIPE
import sys
import tabnanny
import tokenize

try:
    import argparse
except ImportError:
    raise ImportError(
        "check_whitespace.py need Python module argparse introduced in"
        " Python 2.7. It is available in pypi for compatibility."
        " You can install it with this command 'pip install argparse'")

import reindent
from six import StringIO

SKIP_WHITESPACE_CHECK_FILENAME = ".hg/skip_whitespace_check"


def get_parse_error(code):
    """
    Checks code for ambiguous tabs or other basic parsing issues.

    :param code: a string containing a file's worth of Python code
    :returns: a string containing a description of the first parse error encountered,
              or None if the code is ok
    """
    # note that this uses non-public elements from stdlib's tabnanny, because tabnanny
    # is (very frustratingly) written only to be used as a script, but using it that way
    # in this context requires writing temporarily files, running subprocesses, blah blah blah
    code_buffer = StringIO(code)
    try:
        tabnanny.process_tokens(tokenize.generate_tokens(code_buffer.readline))
    except tokenize.TokenError as err:
        return "Could not parse code: %s" % err
    except IndentationError as err:
        return "Indentation error: %s" % err
    except tabnanny.NannyNag as err:
        return "Ambiguous tab at line %d; line is '%s'." % (err.get_lineno(), err.get_line())
    return None


def clean_diff_line_for_python_bug_2142(diff_line):
    if diff_line.endswith("\n"):
        return diff_line
    else:
        return diff_line + "\n\\ No newline at end of file\n"


def get_correct_indentation_diff(code, filename):
    """
    Generate a diff to make code correctly indented.

    :param code: a string containing a file's worth of Python code
    :param filename: the filename being considered (used in diff generation only)
    :returns: a unified diff to make code correctly indented, or
              None if code is already correctedly indented
    """
    code_buffer = StringIO(code)
    output_buffer = StringIO()
    reindenter = reindent.Reindenter(code_buffer)
    reindenter.run()
    reindenter.write(output_buffer)
    reindent_output = output_buffer.getvalue()
    output_buffer.close()
    if code != reindent_output:
        diff_generator = difflib.unified_diff(code.splitlines(True), reindent_output.splitlines(True),
                                              fromfile=filename, tofile=filename + " (reindented)")
        # work around http://bugs.python.org/issue2142
        diff_tuple = map(clean_diff_line_for_python_bug_2142, diff_generator)
        diff = "".join(diff_tuple)
        return diff
    else:
        return None


def is_merge():
    parent2 = os.environ.get("HG_PARENT2", None)
    return parent2 is not None and len(parent2) > 0


def parent_commit():
    parent1 = os.environ.get("HG_PARENT1", None)
    return parent1


class MercurialRuntimeError(Exception):
    pass


def run_mercurial_command(hg_command):
    hg_executable = os.environ.get("HG", "hg")
    hg_command_tuple = hg_command.split()
    hg_command_tuple.insert(0, hg_executable)
    # If you install your own mercurial version in your home
    # hg_executable does not always have execution permission.
    if not os.access(hg_executable, os.X_OK):
        hg_command_tuple.insert(0, sys.executable)
    try:
        hg_subprocess = Popen(hg_command_tuple, stdout=PIPE, stderr=PIPE)
    except OSError as e:
        print("Can't find the hg executable!", file=sys.stderr)
        print(e)
        sys.exit(1)

    hg_out, hg_err = hg_subprocess.communicate()
    if len(hg_err) > 0:
        raise MercurialRuntimeError(hg_err)
    return hg_out


def parse_stdout_filelist(hg_out_filelist):
    files = hg_out_filelist.split()
    files = [f.strip(string.whitespace + "'") for f in files]
    files = list(filter(operator.truth, files))  # get rid of empty entries
    return files


def changed_files():
    hg_out = run_mercurial_command("tip --template '{file_mods}'")
    return parse_stdout_filelist(hg_out)


def added_files():
    hg_out = run_mercurial_command("tip --template '{file_adds}'")
    return parse_stdout_filelist(hg_out)


def is_python_file(filename):
    return filename.endswith(".py")


def get_file_contents(filename, revision="tip"):
    hg_out = run_mercurial_command("cat -r %s %s" % (revision, filename))
    return hg_out


def save_commit_message(filename):
    commit_message = run_mercurial_command("tip --template '{desc}'")
    with open(filename, "w") as save_file:
        save_file.write(commit_message)


def save_diffs(diffs, filename):
    diff = "\n\n".join(diffs)
    with open(filename, "w") as diff_file:
        diff_file.write(diff)


def should_skip_commit():
    if not os.path.exists(SKIP_WHITESPACE_CHECK_FILENAME):
        return False
    with open(SKIP_WHITESPACE_CHECK_FILENAME, "r") as whitespace_check_file:
        whitespace_check_changeset = whitespace_check_file.read()
    return whitespace_check_changeset == parent_commit()


def save_skip_next_commit():
    with open(SKIP_WHITESPACE_CHECK_FILENAME, "w") as whitespace_check_file:
        whitespace_check_file.write(parent_commit())


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Pretxncommit hook for Mercurial to check for whitespace issues")
    parser.add_argument("-n", "--no-indentation",
                        action="store_const",
                        default=False,
                        const=True,
                        help="don't check indentation, just basic parsing"
                       )
    parser.add_argument("-i", "--incremental",
                        action="store_const",
                        default=False,
                        const=True,
                        help="only block on newly introduced indentation problems; ignore all others"
                       )
    parser.add_argument("-p", "--incremental-with-patch",
                        action="store_const",
                        default=False,
                        const=True,
                        help="only block on newly introduced indentation problems; propose a patch for all others"
                       )
    parser.add_argument("-s", "--skip-after-failure",
                        action="store_const",
                        default=False,
                        const=True,
                        help="when this pre-commit hook fails, don't run it on the next commit; "
                             "this lets you check in your changes and then check in "
                             "any necessary whitespace changes in the subsequent commit"
                       )
    args = parser.parse_args(argv)

    # -i and -s are incompatible; if you skip checking, you end up with a not-correctly-indented
    # file, which -i then causes you to ignore!
    if args.skip_after_failure and args.incremental:
        print("*** check whitespace hook misconfigured! -i and -s are incompatible.", file=sys.stderr)
        return 1

    if is_merge():
        # don't inspect merges: (a) they're complex and (b) they don't really introduce new code
        return 0

    if args.skip_after_failure and should_skip_commit():
        # we're set up to skip this one, so skip it, but
        # first, make sure we don't skip the next one as well :)
        os.remove(SKIP_WHITESPACE_CHECK_FILENAME)
        return 0

    block_commit = False

    diffs = []

    added_filenames = added_files()
    changed_filenames = changed_files()

    for filename in filter(is_python_file, added_filenames + changed_filenames):
        code = get_file_contents(filename)
        parse_error = get_parse_error(code)
        if parse_error is not None:
            print("*** %s has parse error: %s" % (filename, parse_error), file=sys.stderr)
            block_commit = True
        else:
            # parsing succeeded, it is safe to check indentation
            if not args.no_indentation:
                was_clean = None  # unknown
                # only calculate was_clean if it will matter to us
                if args.incremental or args.incremental_with_patch:
                    if filename in changed_filenames:
                        old_file_contents = get_file_contents(filename, revision=parent_commit())
                        was_clean = get_correct_indentation_diff(old_file_contents, "") is None
                    else:
                        was_clean = True  # by default -- it was newly added and thus had no prior problems

                check_indentation = was_clean or not args.incremental
                if check_indentation:
                    indentation_diff = get_correct_indentation_diff(code, filename)
                    if indentation_diff is not None:
                        if was_clean or not args.incremental_with_patch:
                            block_commit = True
                        diffs.append(indentation_diff)
                        print("%s is not correctly indented" % filename, file=sys.stderr)

    if len(diffs) > 0:
        diffs_filename = ".hg/indentation_fixes.patch"
        save_diffs(diffs, diffs_filename)
        print("*** To fix all indentation issues, run: cd `hg root` && patch -p0 < %s" % diffs_filename, file=sys.stderr)

    if block_commit:
        save_filename = ".hg/commit_message.saved"
        save_commit_message(save_filename)
        print("*** Commit message saved to %s" % save_filename, file=sys.stderr)

        if args.skip_after_failure:
            save_skip_next_commit()
            print("*** Next commit attempt will not be checked. To change this, rm %s" % SKIP_WHITESPACE_CHECK_FILENAME, file=sys.stderr)

    return int(block_commit)


if __name__ == '__main__':
    sys.exit(main())
