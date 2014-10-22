#!/usr/bin/env python


__authors__ = "Ian Goodfellow"
__contact__ = "goodfeli@iro"


"""
Usage: run the script from the Theano directory and follow the prompts.

A script for correcting formatting in theano code.
When run, it will suggest corrections and ask whether to apply each
of them.

Currently it just checks that there is a space at the start of each
comment, but eventually I hope it could do PEP8 screening
automatically.

"""

import stat
import os
import shutil
import re
import subprocess as sp


rules = []

def run_shell_command(cmd):
        """ Runs cmd as a shell command.
            Waits for it to finish executing, then returns all output
            printed to standard error and standard out, and the return code
        """
        child = sp.Popen(cmd, shell = True, stdout = sp.PIPE, stderr = sp.STDOUT)
        output = child.communicate()[0]
        rc = child.returncode
        return output, rc

def get_choice( choice_to_explanation ):
    """
        choice_to_explanation: a dictionary mapping possible user responses
                               to strings describing what that response will
                               cause the script to do
    """
    d = choice_to_explanation

    for key in d:
        print '\t'+key + ': '+d[key]
    prompt = '/'.join(d.keys())+'? '

    first = True
    choice = ''
    while first or choice not in d.keys():
        if not first:
            print 'unrecognized choice'
        first = False
        choice = raw_input(prompt)
    return choice


def process_file(filepath):

    # load the file
    f = open(filepath)
    lines = f.readlines()
    f.close()

    # remember what the file was like before changing it
    orig_lines = list(lines)

    print 'process '+filepath+'?'

    choice = get_choice({'y' : 'process the file',
                         'n' : 'do not process the file',
                         'q' : 'do not process the file, and quit'})

    if choice == 'n':
        return

    if choice == 'q':
        quit()


    # run all the rules on the file
    for rule in rules:
        rule.process(lines)

    # Only save the file / git add it if some changes were made
    if len(orig_lines) != len(lines) or False in \
        [ orig == current for orig, current in zip(lines,orig_lines)]:
        # get the file permissions of the current file
        orig_mod = os.stat(filepath)[stat.ST_MODE]
        if not isinstance(orig_mod, int):
            raise AssertionError("Expected to get an int but got %s"
                    "of type %s" % (orig_mod, type(orig_mod)))
        # make a backup in case writing fails
        shutil.move(filepath,filepath+'.bak')
        success = False
        try:
            f = open(filepath,'w')
            f.write(''.join(lines))
            f.close()
            os.chmod(filepath,orig_mod)
            success = True
        except:
            # if any error occurred, restore the backup
            shutil.move(filepath+'.bak',filepath)
            raise
        # if the write succeeded, remove the backup
        if success:
            os.remove(filepath+'.bak')

        print 'Done processing this file. Run git add on it?'

        choice = get_choice( {'y' : 'run git add',
                              'n' : 'do not run git add'} )

        if choice == 'y':
            output, rc = run_shell_command('git add '+filepath)
            print 'git return code: '+str(rc)
            print  'git output:\n'+output


def process_directory(directory):

    names = os.listdir(directory)

    for name in names:
        full = directory + '/' + name
        if name.endswith('.py'):
            process_file(full)
        elif os.path.isdir(full):
            process_directory(full)

class Rule:

    def process(self, lines):
        raise NotImplementedError()


class CommentSpaceRule:

    def __init__(self):

        self.prog = re.compile("#[A-Za-z0-9]")

    def process(self, lines):

        print "Applying the rule that single-line comments " + \
                "should start with a space."
        print "Good: # comment"
        print "Bad: # comment"

        for i, line in enumerate(lines):
            match = self.prog.search(line)
            if match:
                print 'Change'
                print line
                start = match.start()
                fixed = line[0:start+1]+' '+line[start+1:]
                print 'to'
                print fixed
                x = get_choice({'y' : 'make the change',
                    'n' : 'do not make the change',
                    'q' : 'quit the script and discard all changes so far to this file'})
                if x == 'q':
                    quit()
                if x == 'y':
                    lines[i] = fixed

rules.append(CommentSpaceRule())

if __name__ == '__main__':
    process_directory('.')
