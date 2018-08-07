from typing import *

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
import shutil
import functools

from io import StringIO

LOADERS = {
    ".bz2": bz2.open,
    ".xz": lzma.open,
    ".lzma": lzma.open,
    ".gz": gzip.open,
    ".gzip": gzip.open,
}

class suppress_exit:
    def __init__(self, obj):
        self.obj = obj
    
    def __enter__(self, *args, **kwargs):
        return self.obj

    def __exit__(self, *args, **kwargs):
        return self

def smart_open(filename, *args, **kwargs):
    if filename == '-':
        if len(args) == 0 or arg[0].startswith('r'):
            return suppress_exit(sys.stdin)
        if len(args) > 0 and args[0].starswith('w'):
            return suppress_exit(sys.stdout)
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
    return operator.attrgetter(dest)

def parser_add_parallelism(parser, dest="nprocs"):
    parser.add_argument("--nprocs", dest=dest, nargs='?', type=int,
                     default=multiprocessing.cpu_count(),
                     help="The level of parallelism, or the number of processors/cores. Default: %(default)s")
    return operator.attrgetter(dest)

def parser_add_salento_home(parser, dest="salento_home"):
    parser.add_argument("--salento-home", dest=dest, default=os.environ.get('SALENTO_HOME', None),
        required=os.environ.get('SALENTO_HOME', None) is None,
        help="The directory where the salento repository is located (defaults to $SALENTO_HOME). Default: %(default)r")
    return operator.attrgetter(dest)


def get_home():
    return os.path.abspath(os.path.dirname(sys.argv[0]))

def wc_binary_path():
    return os.path.join(get_home(), 'salento-wc.sh')

def quote(msg, *args):
    try:
        return msg % tuple(shlex.quote(f) for f in args)
    except TypeError as e:
        raise ValueError(str(e), msg, args)

def run(cmd, *args, silent=True, echo=False, dry_run=False, stdout=None, stderr=None):
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
            if stdout is None:
                stdout = fd
            if stderr is None:
                stderr = fd

        if stdout is not None:
            kwargs['stdout'] = fd
        if stderr is not None:
            kwargs['stderr'] = fd

        return subprocess.call(cmd, shell=True, **kwargs)
    finally:
        if fd is not None:
            fd.close()

def run_or_cleanup(cmd, outfile, print_err=False):
    try:
        err = StringIO()
        if run(cmd, silent=False, stderr=err) != 0:
            if print_err:
                shutil.copyfileobj(err, sys.stderr)
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

def delete_dir(dirname):
    try:
        shutil.rmtree(dirname)
    except FileNotFoundError:
        pass

def delete(filename):
    try:
        delete_dir(filename)
    except NotADirectoryError:
        delete_file(filename)


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

def parse_slice(expr) -> slice:
    """
    Parses standard Python slices from a string.

    The usual expressions apply:

        >>> [1,2,3][parse_slice("0")]
        [1]
        >>> [1,2,3][parse_slice("1")]
        [2]
        >>> [1,2,3][parse_slice("1:")]
        [2, 3]
        >>> [1,2,3][parse_slice("1:-1")]
        [2]
        >>> parse_slice(":-1")
        slice(None, -1, None)

    Parsing errors are flagged as `ValueError`s:

        >>> [1,2,3][parse_slice("1:-1  k")]
        Traceback (most recent call last):
        ...
        ValueError: invalid literal for int() with base 10: '-1  k'
    """
    def to_piece(s):
        return int(s) if len(s) > 0 else None
    pieces = list(map(to_piece, expr.split(':')))
    if len(pieces) == 1:
        return slice(pieces[0], pieces[0] + 1)
    else:
        return slice(*pieces)

def get_slice_indices(slc, elems):
    """
    Coverts a slice into a list of objects

        >>> list(get_slice_indices(slice(None, None, None), [10,65,3,0]))
        [0, 1, 2, 3]

    """
    return range(*slc.indices(len(elems)))

def from_slice(slc, elems):
    """
    Coverts a slice into a list of objects

        >>> from_slice(slice(None, None, None), [1,2,3])
        [1, 2, 3]

    """
    return list(map(elems.__getitem__, get_slice_indices(slc, elems)))

def split_exts(filename):
    """
    Instead of only separating the first extension, like `os.splitext` does, separate all:

        >>> split_exts('foo.bar.bz')
        ('foo', '.bar.bz')

    `split_exts` works as `os.path.splitext` when there is only one extension:

        >>> split_exts('foo.bar')
        ('foo', '.bar')

    As usual, it will return an empty string when there is no extension:

        >>> split_exts('foo')
        ('foo', '')

    """
    filename, ext = os.path.splitext(filename)
    result = ext
    while ext != '':
        filename, ext = os.path.splitext(filename)
        result = ext + result
    return filename, result

def skip_n(iterable, count):
    it = iter(iterable)
    # Skip the first n elements
    try:
        for _ in range(count):
            next(it)
    except StopIteration:
        return
    # Return the rest
    yield from it

def take_n(iterable, count):
    """
    >>> it = iter([1,2,3,4,5])
    >>> list(take_n(it, 2))
    [1, 2]
    >>> list(take_n(it, 2))
    [3, 4]
    >>> list(take_n(it, 2))
    [5]
    >>> list(take_n(it, 10))
    []
    """
    it = iter(iterable)
    while count > 0:
        yield next(it)
        count -= 1

def partition_iter(iterable, counts):
    """
    >>> list(map(list, partition_iter([1,2,3,4,5,6,7], [2, 5])))
    [[1, 2], [3, 4, 5, 6, 7]]
    """
    it = iter(iterable)
    for count in counts:
        yield take_n(it, count)

def cons_last(iterable, elem):
    yield from iterable
    yield elem

def memoize(fun):
    return functools.lru_cache(maxsize=None)(fun)

def as_list(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return list(f(*args, **kwargs))
    return wrapper

def parse_ranges(expr:str) -> List[slice]:
    expr = expr.strip()
    if expr == '' or expr == '*':
        return [common.parse_slice(":")]
    return list(map(common.parse_slice, expr.split(",")))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
