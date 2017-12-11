import sys
import os
import subprocess
import collections
import errno
import json
import multiprocessing
import concurrent.futures
import glob
import itertools

def parser_add_input_files(parser):
    parser.add_argument("-f", dest="infiles", nargs='+', type=str,
                     default=[], help="A file of the Salento Dataset format.")
    parser.add_argument("-i", dest="use_stdin",
                     help="Read filenames from input.",
                     action="store_true")
    parser.add_argument("-d", dest="dir", nargs='?', type=str,
                     default=None, help="A directory containing Salento JSON Package. Default: standard input.")
    
    def parser_get_input_files(args):
        infiles = list(args.infiles)

        if args.use_stdin:
            infiles = itertools.chain(infiles, (x.strip() for x in sys.stdin if not x.strip().startswith("#")))

        if args.dir is not None:
            infiles = itertools.chain(infiles, find_sal(args.dir))
        
        infiles = list(infiles)
        infiles.sort()
        return infiles

    return parser_get_input_files

def parser_add_wc_binary(parser, dest="wc_binary"):
    parser.add_argument("--salento-wc-exec", nargs="?",
                    default=wc_binary_path(),
                    dest=dest,
                    help="The `salento-wc.py` binary. Default: %(default)s.")
    return lambda args: getattr(args, dest)

def parser_add_parallelism(parser, dest="nprocs"):
    parser.add_argument("--nprocs", dest=dest, nargs='?', type=int,
                     default=multiprocessing.cpu_count(), help="The maximum number of parallel word counts. Default: %(default)s.")
    return lambda x: getattr(x, dest)

def get_home():
    return os.path.abspath(os.path.dirname(sys.argv[0]))

def wc_binary_path():
    return os.path.join(get_home(), 'salento-wc.sh')

def run(cmd, silent=True):
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

def command(label, infile, outfile, cmd, show_command=False):
    if show_command:
        print(cmd)
    else:
        print(label + " " + outfile)
    try:
        if run(cmd) != 0:
            print("ERROR: " + cmd)
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

def find_sal(dirname):
    return itertools.chain(
        find_files(dirname, "*.sal"),
        find_files(dirname, "*.sal.bz2")
    )

def word_freq(program, filename):
    target_file = filename + ".wc"
    if not os.path.exists(target_file):
        if subprocess.call(program + " " + filename + " > " + target_file, shell=True) != 0:
            delete_file(target_file)
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
