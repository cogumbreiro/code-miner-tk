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
    parser.add_argument("--aggregator", "-a", default="sequence", help="The aggregator to run. Default: %(default)r")
    parser.add_argument("--log", action="store_true", help="Save output to a log file.")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually run any program, just print the commands.")
    parser.add_argument("--profile", help="Runs Salento behind cProfiler. PROFILE the profiling filename.")
    parser.add_argument("--echo", help="Shows the executed commands.", action="store_true")
    parser.add_argument("filenames", nargs="+", default=[],
                    help="The JSON filename we are processing.")
    get_nprocs = common.parser_add_parallelism(parser)
    get_salento = common.parser_add_salento_home(parser)
    args = parser.parse_args()

    if args.profile:
        prof = "-m cProfile -o " + shlex.quote(args.profile)
    else:
        prof = ""

    script = os.path.join(get_salento(args), "src/main/python/salento/aggregators/%s_aggregator.py" % args.aggregator)
    nprocs = get_nprocs(args)
    with common.finish(concurrent.futures.ThreadPoolExecutor(max_workers=nprocs)) as executor:
        for fname in args.filenames:
            @executor.submit
            def run(script=script, data_dir=args.data_dir, fname=fname):
                if args.log:
                    extra = " > " + shlex.quote(fname + ".log")
                else:
                    extra = ""
                common.run("python3 " + prof + " %s --model_dir %s --data_file %s" + extra, script, data_dir, fname, echo=args.echo, silent=False, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
