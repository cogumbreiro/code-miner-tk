#!/usr/bin/env python3
import copy
import math
import tempfile
import json
import os.path
import concurrent.futures
import glob
import shlex

try:
    import common
except ImportError:
    import sys
    import os
    from os import path
    home = path.abspath(path.dirname(sys.argv[0]))
    sys.path.append(path.join(home, "src"))

import common

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Runs a Salento aggregator for each file given as input.")
    parser.add_argument("-d", dest="data_dir", type=str,
                    required=not os.path.exists("save"),
                    default="save", help="The default Tensorflow model directory. Default: %(default)r")
    parser.add_argument("-f", dest="filenames", nargs="+", default=[],
                    help="The JSON filename we are processing.")
    parser.add_argument("--salento-home", dest="salento_home", default=os.environ.get('SALENTO_HOME', None),
        required=os.environ.get('SALENTO_HOME', None) is None,
        help="The directory where the salento repository is located (defaults to $SALENTO_HOME). Default: %(default)r")
    
    parser.add_argument("--aggregator", default="sequence", help="The aggregator to run. Default: %(default)r")
    parser.add_argument("--log", action="store_true", help="Save output to a log file.")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually run any program, just print the commands.")
    get_nprocs = common.parser_add_parallelism(parser)
    args = parser.parse_args()

    script = os.path.join(args.salento_home, "src/main/python/salento/aggregators/%s_aggregator.py" % args.aggregator)
    nprocs = get_nprocs(args)
    with common.finish(concurrent.futures.ThreadPoolExecutor(max_workers=nprocs)) as executor:
        for fname in args.filenames:
            @executor.submit
            def run(script=script, data_dir=args.data_dir, fname=fname):
                if args.log:
                    extra = " > " + shlex.quote(fname + ".log")
                else:
                    extra = ""
                common.run("python3 %s --model_dir %s --data_file %s" + extra, script, data_dir, fname, silent=False, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
