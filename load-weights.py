#!/usr/bin/env python3
#!/usr/bin/env python3
import sys
import os
import os.path
import warnings
import cmd
import argparse
import collections
import numpy as np
from operator import *
from typing import *
import json

# Shut up Tensorflow
if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

from argparse import Namespace
from replui import argparse_cmd, parse_ranges, repl_format, REPLExit
from salui import make_app, ASequence, load_state_anomaly, StateAnomalyFilter
from collections import Counter
from statedist import CallDistNorm, TermDistNorm

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input data file')
    parser.add_argument('out', help='output data file')
    parser.add_argument('--resume-pid', type=int, default=0)
    parser.add_argument('--resume-sid', type=int, default=-1)
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--print-pid', action='store_true')
    parser.add_argument('--print-sid', action='store_true')
    parser.add_argument('--dirname', '-d', default="save",
                        help='directory to load model from')
    args = parser.parse_args()

    with make_app(args.filename, args.dirname) as app:
        app.init()
        filter_anomalies = StateAnomalyFilter(threshold=0.2)
        seqs = (seq for pkg in app.pkgs for seq in pkg)
        mode = 'a+' if args.append else 'w'
        with open(args.out, mode) as fp:
            pid = 0
            sid = 0
            try:
                for pid, pkg in enumerate(app.pkgs):
                    sid = 0
                    if args.resume_pid > pid:
                        continue
                    if args.print_pid:
                        print('--resume-pid', pid, '--resume-sid', sid)
                    for sid, seq in enumerate(seqs):
                        if args.resume_pid == pid and args.resume_sid > sid:
                            continue
                        if args.print_sid:
                            print('--resume-pid', pid, '--resume-sid', sid)
                        for row in filter_anomalies(seq):
                            print(json.dumps(row), file=fp)
            except KeyboardInterrupt:
                print('--resume-pid', pid, '--resume-sid', sid)

        #db = load_state_anomaly(seqs,
        #    state_count=3,
        #    threshold=0.2,
        #    #accept_state=lambda x: x.name=="1"
        #)
        #with open(args.out, 'w') as fp:
        #    json.dump(db, fp)

if __name__ == '__main__':
    main()

