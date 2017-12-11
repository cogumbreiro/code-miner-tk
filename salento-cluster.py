#!/usr/bin/env python3

try:
    import salento
except ImportError:
    import sys
    import os
    from os import path
    sys.path.append(path.abspath(path.dirname(sys.argv[0])))
    import salento

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

def repeat(term, count):
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

    get_wc = salento.parser_add_wc_binary(parser)
    get_input_files = salento.parser_add_input_files(parser)
    get_nprocs = salento.parser_add_parallelism(parser)

    parser.add_argument("--nclusters", dest="nclusters", nargs='?', type=int,
                     default=5, help="The number of clusters to use in KMeans. Default: %(default)s.")
    parser.add_argument("--iters", dest="nclusters", nargs='?', type=int,
                     default=5, help="The number of clusters to use in KMeans. Default: %(default)s.")
    parser.add_argument("--include-empty", dest="include_empty", action="store_true",
                     help="By default empty files are ignored; this option will include empty files.")
    args = parser.parse_args()

    infiles = get_input_files(args)
    wc = get_wc(args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=get_nprocs(args)) as executor:
        # 1. Compute the in parallel in the background
        word_freqs = dict(zip(infiles, salento.run_word_freqs(executor, wc, infiles)))
    # 2. Remove empty files
    if not args.include_empty:
        new_infiles = []
        for f in infiles:
            if len(word_freqs[f]) > 0:
                new_infiles.append(f)
            else:
                print("Ignoring empty file:", f, file=sys.stderr)
        infiles = new_infiles

    # 2. Build a TF-IDF
    # simulate a tokenizer
    def analyzer(x):
        for (term, freq) in word_freqs[x].items():
            yield from repeat(term, freq)

    tfidf = TfidfVectorizer(min_df=1, analyzer=analyzer)
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
