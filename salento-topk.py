#!/usr/bin/env python3

import itertools
import sys
import glob
import os
import subprocess
import collections
import errno
import glob
import json
import multiprocessing
import concurrent.futures

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def find_files(dirname, ext):
    return glob.glob(os.path.join(dirname, "**", ext), recursive=True)

def find_sal(dirname):
    return itertools.chain(
        find_files(dirname, "*.sal"),
        find_files(dirname, "*.sal.bz2")
    )

def delete_file(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

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
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clusters a directory containing Salento JSON datasets.")
    parser.add_argument("-f", dest="infiles", nargs='+', type=str,
                     default=[], help="A file of the Salento Dataset format.")
    parser.add_argument("-i", dest="use_stdin",
                     help="Read filenames from input.",
                     action="store_true")
    parser.add_argument("-d", dest="dir", nargs='?', type=str,
                     default=None, help="A directory containing Salento JSON Package. Default: standard input.")
    parser.add_argument("--nclusters", dest="nclusters", nargs='?', type=int,
                     default=5, help="The number of clusters to use in KMeans. Default: %(default)s.")
    parser.add_argument("--iters", dest="nclusters", nargs='?', type=int,
                     default=5, help="The number of clusters to use in KMeans. Default: %(default)s.")
    parser.add_argument("--include-empty", dest="include_empty", action="store_true",
                     help="By default empty files are ignored; this option will include empty files.")
    parser.add_argument("--nprocs", dest="nprocs", nargs='?', type=int,
                     default=multiprocessing.cpu_count(), help="The maximum number of parallel word counts. Default: %(default)s.")
    parser.add_argument("--top-k", dest="top_k", nargs='?', type=int, default=5,
                     help="How many terms to show. Default: %(default)s")

    args = parser.parse_args()

    infiles = list(args.infiles)

    wc = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'salento-wc.sh')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.nprocs) as executor:
        for line in sys.stdin:
            topk = collections.Counter()
            for wf in run_word_freqs(executor, wc, json.loads(line)):
                for (k,v) in wf.items():
                    topk[k] += v
            json.dump(dict(topk.most_common(args.top_k)), sys.stdout)
            print()

if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        pass # No more output, close graciously
