#!/usr/bin/env python3

try:
    import common
except ImportError:
    import sys
    import os
    from os import path
    home = path.abspath(path.dirname(sys.argv[0]))
    sys.path.append(path.join(home, "src"))

import common

import tarfile
import os
import sys
import os.path
import errno
import subprocess
import multiprocessing
import threading
import concurrent.futures
import shlex
import enum

from common import delete_file, finish, run_or_cleanup, parse_file_list, fifo



def target_filename(filename, prefix, extension):
    return os.path.join(prefix, filename + extension)

def quote(msg, *args):
    try:
        return msg % tuple(shlex.quote(f) for f in args)
    except TypeError as e:
        raise ValueError(str(e), msg, args)

class StopExecution(Exception): pass

class Env:
    def __init__(self, args, executor):
        self.args = args
        home_dir = os.path.dirname(sys.argv[0])
        as2sal_bin = os.path.join(home_dir, 'apisan-to-salento.py')
        self.as2sal = shlex.quote(as2sal_bin) + " "
        if args.apisan_translator is not None:
            self.as2sal += "--translator " + args.apisan_translator + " "
        self.tar = tarfile.open(args.infile, "r|*")
        self.apisan = shlex.quote(os.path.join(os.environ['APISAN_HOME'], 'apisan'))
        if args.timeout is not None and args.timeout.strip() != "":
            self.apisan = "timeout " + args.timeout + " " + self.apisan
        self.executor = executor
        self.cancelled = threading.Event()
        skip_files = set(
            [] if args.skip_file is None else parse_file_list(args.skip_file)
        )
        if args.accept_file is None:
            self.reject = skip_files.__contains__
        else:
            accept_files = set(parse_file_list(args.accept_file)) - skip_files
            self.reject = lambda x: x not in accept_files

    def needs_update(self, infile, *outfiles):
        for outfile in outfiles:
            try:
                if os.path.exists(outfile) and (not os.path.exists(infile) or os.path.getctime(outfile) >= os.path.getctime(infile)):
                    return False
            except FileNotFoundError:
                pass # OK, output file not found
        return True

    def run(self, label, infile, outfile, cmd, *args, unless=[]):

        cmd = quote(cmd, *args)

        if not self.needs_update(infile, outfile, *unless):
            # Nothing to do
            return

        if not os.path.exists(infile):
            # Internal error!
            raise StopExecution("Error: file missing: " + infile + "\n\t" + cmd)

        if self.args.verbose:
            print(cmd)
        else:
            print(infile + " -> " + outfile)

        if not run_or_cleanup(cmd, outfile) or not os.path.exists(outfile):
            raise StopExecution("Error: processing file: " + infile + "\n\t" + cmd)
    
    def run_apisan(self, c_fname, as_fname, unless=[]):
        try:
            self.run("APISAN", c_fname, as_fname, self.apisan + " compile %s", c_fname, unless=unless)
            if Run.C not in self.args.keep:
                delete_file(c_fname)
        except StopExecution:
            # Only log when program has not been user-terminated
            if not self.cancelled.is_set() and self.args.log_ignored is not None:
                print(c_fname, file=self.args.log_ignored)
                self.args.log_ignored.flush() # Ensure the filename is written
            raise

    def _spawn(self, func):
        @self.executor.submit
        def task():
            try:
                func()
            except StopExecution as e:
                # Log StopExecution and continue (not terminal errors)
                print(e, file=sys.stderr)

    def cancel(self):
        self.cancelled.set() # mark shared variable as cancelled
        self.executor.cancel_pending() # abort running tasks
        
    def _process(self, tar_info):
        c_fname = tar_info.name
        as_fname = target_filename(c_fname, self.args.prefix, ".as.bz2")
        sal_fname = target_filename(c_fname, self.args.prefix, ".sal.bz2")
        o_file = os.path.splitext(os.path.basename(c_fname))[0] + ".o"

        if not c_fname.endswith(".c") or not tar_info.isfile() or self.reject(c_fname):
            return
        
        if self.args.run == Run.C:
            check_files = [c_fname]
        elif self.args.run == Run.APISAN:
            check_files = [c_fname, as_fname]
        elif self.args.run == Run.SALENTO:
            check_files = [c_fname, as_fname, sal_fname]
        
        if not any(map(os.path.exists, check_files)): # no file exists
            # Extract file
            self.tar.extract(tar_info)
            if not os.path.exists(c_fname):
                print("Error extracting: %r" % c_fname, file=sys.stderr)
                return # nothing else to do
            # Cleanup
            self.tar.members = []

        if self.args.run == Run.APISAN:
            @self._spawn
            def run_salento(c_fname=c_fname, as_fname=as_fname):
                self.run_apisan(c_fname, as_fname)
                delete_file(c_fname)

        elif self.args.run == Run.SALENTO:
            @self._spawn
            def run_apisan():
                self.run_apisan(c_fname, as_fname, unless=[sal_fname])
                self.run("SAN2SAL", as_fname, sal_fname,
                        self.as2sal + "-i %s -o %s", as_fname, sal_fname)
                if not Run.APISAN in self.args.keep:
                    delete_file(as_fname)

    def start(self):
        for x in self.tar:
            self._process(x)


@enum.unique
class Run(enum.Enum):
    C = 1
    APISAN = 2
    SALENTO = 3

    def __repr__(self):
        return self.name

    __str__ = __repr__

    @classmethod
    def from_string(cls, name):
        name = name.upper()
        for run in Run:
            # we don't need to type the whole name for a match
            if run.name.startswith(name):
                return run
        raise ValueError()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Given an archive of C files, generates Salento Package JSON files.")
    parser.add_argument("-v", dest="verbose", help="Verbose output.",
                     action="store_true")
    parser.add_argument("-i", dest="infile", nargs='?', type=str,
                     default="/dev/stdin", help="A filename of the APISAN file format (.as). DEFAULT: '%(default)s'")
    parser.add_argument("-p", dest="prefix", nargs='?', type=str,
                     default="as-out", help="The directory where we are locating. DEFAULT: '%(default)s'")
    parser.add_argument("-d", help="Show commands instead of user friendly label.", dest="debug",
                    action="store_true")
    parser.add_argument("--accept-file", help="A file that contains the C file names to be accepted; one file per line.",
                    metavar="filename",
                    nargs='?', type=argparse.FileType('r'), default=None, dest="accept_file")
    parser.add_argument("--skip-file", help="A file that contains the C file names to be ignored; one file per line.",
                    metavar="filename",
                    nargs='?', type=argparse.FileType('r'), default=None, dest="skip_file")
    parser.add_argument("--log-ignored", help="Log C files which could not be parsed to target filename (which will be appended).",
                    nargs='?', type=argparse.FileType('a'), default=None, dest="log_ignored")
    parser.add_argument("-t", dest="timeout", nargs='?', type=str,
                    default="1h", help="The timeout. DEFAULT: '%(default)s'")
    parser.add_argument("-r", "--run", type=Run.from_string, choices=list(Run), dest='run',
                    default=Run.SALENTO, help="Run until the following state. %(default)s")
    parser.add_argument("-k", "--keep", nargs="+",
                    type=lambda x: map(Run.from_string, x), dest='keep',
                    default=[Run.APISAN, Run.SALENTO], help="Keep the following files, remove any files not listed. %(default)s")
    parser.add_argument("--apisan-translator",
        default=None, help="Set the Apisan-to-Salento translator algorithm. Check `apisan-to-salento.py` for options.")

    get_nprocs = common.parser_add_parallelism(parser)

    args = parser.parse_args()

    with finish(fifo(concurrent.futures.ThreadPoolExecutor(max_workers=get_nprocs(args)), get_nprocs(args))) as executor:
        env = Env(args = args, executor = executor)
        try:
            env.start()
        except KeyboardInterrupt:
            # Cleanup pending commands
            print("Caught a Ctrl-c! Cancelling running tasks.", file=sys.stderr)
            env.cancel()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # User break, it's ok to quit
        pass
