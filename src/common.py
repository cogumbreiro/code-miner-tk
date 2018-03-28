import sys
import os
import operator
import subprocess
import collections
import errno
import json
import multiprocessing
import concurrent.futures
import glob
import shlex
import itertools
import re
import bz2
import lzma
import gzip
import os.path

LOADERS = {
    ".bz2": bz2.open,
    ".xz": lzma.open,
    ".lzma": lzma.open,
    ".gz": gzip.open,
    ".gzip": gzip.open,
}

def smart_open(filename, *args, **kwargs):
    return LOADERS.get(os.path.splitext(filename)[1], open)(filename, *args, **kwargs)

SAL_GLOBS = ("*.sal", "*.sal.bz2")

def parse_file_list(fp):
    pat = re.compile(r'#.*')
    for line in fp:
        line = pat.sub("", line).strip()
        if line != "":
            yield line

def parser_add_input_files(parser):
    parser.add_argument("-f", dest="infiles", nargs='+', type=str,
                     default=[], help="A file of the Salento Dataset format.")
    parser.add_argument("-i", dest="use_stdin",
                     help="Read filenames from input.",
                     action="store_true")
    parser.add_argument("-d", dest="dir", nargs='?', type=str,
                     default=None, help="A directory containing Salento JSON Package.")
    
    def parser_get_input_files(args, lazy=False, globs=SAL_GLOBS):
        infiles = args.infiles

        if args.use_stdin:
            infiles = itertools.chain(infiles, parse_file_list(sys.stdin))

        if args.dir is not None:
            infiles = itertools.chain(infiles, find_any(args.dir, globs))
        
        if lazy:
            return infiles
        
        # Otherwise we sort the files
        infiles = list(infiles)
        infiles.sort()
        return infiles

    return parser_get_input_files

def parser_add_wc_binary(parser, dest="wc_binary"):
    parser.add_argument("--salento-wc-exec", nargs="?",
                    default=wc_binary_path(),
                    dest=dest,
                    help="The `salento-wc.py` binary. Default: %(default)s.")
    return operator.itemgetter(dest)

def parser_add_parallelism(parser, dest="nprocs"):
    parser.add_argument("--nprocs", dest=dest, nargs='?', type=int,
                     default=multiprocessing.cpu_count(),
                     help="The level of parallelism, or the number of processors/cores. Default: %(default)s")
    return operator.itemgetter(dest)

def parser_add_salento_home(parser, dest="salento_home"):
    parser.add_argument("--salento-home", dest=dest, default=os.environ.get('SALENTO_HOME', None),
        required=os.environ.get('SALENTO_HOME', None) is None,
        help="The directory where the salento repository is located (defaults to $SALENTO_HOME). Default: %(default)r")
    return operator.itemgetter(dest)


def get_home():
    return os.path.abspath(os.path.dirname(sys.argv[0]))

def wc_binary_path():
    return os.path.join(get_home(), 'salento-wc.sh')

def quote(msg, *args):
    try:
        return msg % tuple(shlex.quote(f) for f in args)
    except TypeError as e:
        raise ValueError(str(e), msg, args)

def run(cmd, *args, silent=True, echo=False, dry_run=False):
    cmd = quote(cmd, *args)
    if echo:
        print(cmd)
    if dry_run:
        return
    kwargs = dict()
    fd = None
    try:
        if silent:
            fd = open(os.devnull, 'w')
            kwargs['stdout'] = fd
            kwargs['stderr'] = fd
        return subprocess.call(cmd, shell=True, **kwargs)
    finally:
        if fd is not None:
            fd.close()

def run_or_cleanup(cmd, outfile):
    try:
        if run(cmd) != 0:
            print("ERROR: " + cmd, file=sys.stderr)
            delete_file(outfile)
            return False
    except:
        delete_file(outfile)
        raise
    return True

def delete_file(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def find_files(dirname, ext):
    return glob.glob(os.path.join(dirname, "**", ext), recursive=True)

def find_any(dirname, patterns):
    result = ()
    for pattern in patterns:
        result = itertools.chain(result, find_files(dirname, pattern))
    return result

def word_freq(program, filename):
    target_file = filename + ".wc"
    if not os.path.exists(target_file):
        run_or_cleanup(program + " " + filename + " > " + target_file, target_file)
    with open(target_file) as fp:
        for line in fp:
            line = line.strip()
            if line == "": continue
            freq, term = line.split(" ")
            yield (term, int(freq))

def run_word_freqs(executor, wc, infiles):
    # Create a list to force all futures to be spawned
    futs = [executor.submit(lambda: dict(word_freq(wc, f))) for f in infiles]
    for f in futs:
        yield f.result()

class fifo:
    def __init__(self, executor, count):
        self.executor = executor
        self.pending = []
        self.count = count

    def submit(self, *args):
        while len(self.pending) >= self.count:
            self.pending[0].result()
            del self.pending[0]
        # We can now submit the task
        fut = self.executor.submit(*args)
        self.pending.append(fut)
        return fut

    def __enter__(self):
        self.executor.__enter__()
        return self
    
    def __exit__(self, *args):
        del self.pending
        self.executor.__exit__(*args)


class finish:
    def __init__(self, executor, accumulator=lambda x: x, steps=100):
        self.executor = executor
        self.accumulate = accumulator
        self.pending = []
        self.count = 0
        self.steps = steps

    def garbage_collect(self):
        # Garbage collect
        if self.count % self.steps == 0:
            self.count = 0 # reset counter
            to_remove = []
            for fut in filter(lambda x: x.done(), self.pending):
                self.accumulate(fut.result())
                to_remove.append(fut)
            for x in to_remove:
                self.pending.remove(x)

    def submit(self, *args):
        fut = self.executor.submit(*args)
        self.pending.append(fut)
        self.garbage_collect()
        self.count += 1
        return fut

    def shutdown(self, *args, **kwargs):
        return self.executor.shutdown(*args, **kwargs)

    def map(self, *args, **kwargs):
        return self.executor.map(*args, **kwargs)

    def cancel_pending(self):
        for x in self.pending:
            x.cancel()

    def __enter__(self):
        self.executor.__enter__()
        return self
    
    def __exit__(self, *args):
        for fut in self.pending:
            self.accumulate(fut.result())
        del self.pending
        del self.accumulate
        self.executor.__exit__(*args)


def human_size(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
    """ Returns a human readable string reprentation of bytes"""
    return str(bytes) + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

