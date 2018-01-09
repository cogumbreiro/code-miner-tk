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
import concurrent.futures
import shlex
import enum

from common import delete_file, finish, run_or_cleanup, parse_file_list, fifo

def command(label, infile, outfile, cmd, show_command=False, silent=False):
    if not silent:
        if show_command:
            print(cmd)
        else:
            print(label + " " + outfile)
    if run_or_cleanup(cmd, outfile):
        file_exists = os.path.exists(outfile)
        if not file_exists:
            print("Error processing: %r" % infile, cmd, file=sys.stderr)
        return file_exists
    else:
        return False


def target_filename(filename, prefix, extension):
    return os.path.join(prefix, filename + extension)

def quote(msg, *args):
    try:
        return msg % tuple(shlex.quote(f) for f in args)
    except TypeError as e:
        raise ValueError(str(e), msg, args)

def processor(args, as2sal, ignore, tar, apisan, executor):

    def do_run(tar_info):
        c_fname = tar_info.name
        as_fname = target_filename(c_fname, args.prefix, ".as")
        as_bz_fname = target_filename(c_fname, args.prefix, ".as.bz2")
        sal_fname = target_filename(c_fname, args.prefix, ".sal")
        sal_bz_fname = target_filename(c_fname, args.prefix, ".sal.bz2")
        o_file = os.path.splitext(os.path.basename(c_fname))[0] + ".o"
        if not c_fname.endswith(".c") or not tar_info.isfile():
            return
        if c_fname in ignore:
            if args.verbose: print("SKIP " + c_fname)
            return

        if args.run == Run.C or (not os.path.exists(as_fname) and not os.path.exists(sal_bz_fname)):
            print("TAR " + c_fname)
            # Extract file
            tar.extract(tar_info)
            if not os.path.exists(c_fname):
                print("Error extracting: %r" % c_fname, file=sys.stderr)
                return # nothing else to do
            # Cleanup
            tar.members = []
        if args.run == Run.C:
            return

        @executor.submit
        def continuation(): # The rest should be scheduled in parallel
            if not os.path.exists(as_fname) and not os.path.exists(sal_bz_fname):
                # Compile file
                try:
                    if command("APISAN", c_fname, as_fname, quote(apisan + " compile %s", c_fname), args.debug):
                        # Remove filename on success
                        if Run.C not in args.keep:
                            delete_file(c_fname)
                    else:
                        if args.log_ignored is not None:
                            print(c_fname, file=args.log_ignored)
                            args.log_ignored.flush() # Ensure the filename is written
                        return
                finally:
                    # This file is never needed for debugging
                    delete_file(o_file)

            if args.run == Run.APISAN:
                return

            if os.path.exists(as_fname) and not os.path.exists(sal_bz_fname):
                if command("SAN2SAL", as_fname, sal_bz_fname,
                        quote("python3 %s -i %s | bzip2 > %s", as2sal, as_fname, sal_bz_fname), args.debug):
                    if Run.APISAN in args.keep:
                        if not command("BZ2", as_fname, as_fname + ".bz2", quote("bzip2 %s", as_fname), args.debug):
                            delete_file(as_fname + ".bz2")
                    else:
                        delete_file(as_fname)

            if os.path.exists(sal_bz_fname):
                if args.debug:
                    print("# DONE " + sal_bz_fname)
                elif args.verbose:
                    print("DONE " + sal_bz_fname)

    return do_run

@enum.unique
class Run(enum.Enum):
    C = 1
    APISAN = 2
    SALENTO = 3

    def __repr__(self):
        return self.name

    __str__ = __repr__

    @classmethod
    def from_string(cls, s):
        try:
            return cls[s.upper()]
        except KeyError:
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
                    default=[Run.SALENTO], help="Keep the following files, remove any files not listed. %(default)s")
    get_nprocs = common.parser_add_parallelism(parser)

    args = parser.parse_args()
    apisan = os.path.join(os.environ['APISAN_HOME'], 'apisan')
    if args.timeout is not None and args.timeout.strip() != "":
        apisan = "timeout " + args.timeout + " " + apisan
    tar = tarfile.open(args.infile, "r|*")
    as2sal = os.path.join(os.path.dirname(sys.argv[0]), 'apisan-to-salento.py')
    skip_files = set(
        parse_file_list(args.skip_file) if args.skip_file is not None else []
    )

    with finish(fifo(concurrent.futures.ThreadPoolExecutor(max_workers=get_nprocs(args)), get_nprocs(args))) as executor:
        do_run = processor(args, as2sal, skip_files, tar, apisan, executor)
        try:
            for x in tar:
                do_run(x)

        except KeyboardInterrupt:
            # Cleanup pending commands
            print("Caught a Ctrl-c! Cancelling running tasks.", file=sys.stderr)
            executor.cancel_pending()
            raise

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # User break, it's ok to quit
        pass
