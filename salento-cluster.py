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

def word_freq(executor, program, filename):
    target_file = filename + ".wc"
    if not os.path.exists(target_file):
        fut = executor.submit(lambda: subprocess.call(program + " " + filename + " > " + target_file, shell=True) != 0)
        if fut.result():
            delete_file(target_file)
    with open(target_file) as fp:
        for line in fp:
            line = line.strip()
            if line == "": continue
            freq, term = line.split(" ")
            yield (term, int(freq))

class analyzer:
    def __init__(self, executor, wc):
        self.executor = executor
        self.wc = wc

    def __call__(self, f):
        for term, count in word_freq(self.executor, self.wc, f):
            for _ in range(count):
                yield term

def do_cluster(infiles, km):
    cluster_ids = list([] for x in range(km.n_clusters))
    for doc_id, cluster_id in enumerate(km.labels_.tolist()):
        fname = infiles[doc_id]
        cluster_ids[cluster_id].append(fname)
    cluster_ids.sort(key=list.__len__, reverse=True)
    return cluster_ids

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
    
    args = parser.parse_args()

    infiles = list(args.infiles)

    if args.use_stdin:
        infiles = itertools.chain(infiles, (x.strip() for x in sys.stdin if not x.strip().startswith("#")))

    if args.dir is not None:
        infiles = itertools.chain(infiles, find_sal(args.dir))
    
    infiles = list(infiles)
    infiles.sort()
    wc = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'salento-wc.sh')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.nprocs) as executor:
        if not args.include_empty:
            new_infiles = []
            for f in infiles:
                if len(dict(word_freq(executor, wc, f))) > 0:
                    new_infiles.append(f)
                else:
                    print("Ignoring empty file:", f, file=sys.stderr)
                    
            infiles = new_infiles

        tfidf = TfidfVectorizer(min_df=1, analyzer=analyzer(executor, wc))
        tfidf_matrix = tfidf.fit_transform(infiles)
        km = KMeans(n_clusters=args.nclusters, init='k-means++', max_iter=100, n_init=1,
                    verbose=False)
        km.fit(tfidf_matrix)
        for v in do_cluster(infiles, km):
            print(json.dumps(v))

if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        pass # No more output, close graciously
