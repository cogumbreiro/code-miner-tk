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
from replui import argparse_cmd, repl_format, REPLExit
from salui import make_app, get_state_probs, get_anomalous_states
import sal

# Because we use this in a tight loop, we handle the raw JSON objects directly
class StateAnomalyFilter:

    def __init__(self, threshold, accept_state=lambda x:True):
        self.visited = set()
        self.threshold = threshold
        self.accept_state = accept_state

    def __call__(self, app, pkg_spec, seq):
        calls = sal.get_calls(seq=seq)
        for evt, call in zip(calls, get_state_probs(app, pkg_spec, calls)):
            for idx, st in get_anomalous_states(call, self.threshold):
                if not self.accept_state(st):
                    continue
                loc = sal.get_call_location(evt)
                cid = (loc, call.name, idx)
                if cid in self.visited:
                    continue
                self.visited.add(cid)
                yield call.name, idx, st.name

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

    # We do not need to cache queries because we are running as batch
    with make_app(args.filename, args.dirname, cache=None) as app:
        app.init()
        filter_anomalies = StateAnomalyFilter(threshold=0.2)
        mode = 'a+' if args.append else 'w'
        with open(args.out, mode) as fp:
            pid = 0
            sid = 0
            try:
                pkgs = sal.get_packages(app.dataset)
                for p, pkg in enumerate(pkgs[args.resume_pid:], args.resume_pid):
                    pid = p
                    seqs = sal.get_sequences(pkg)
                    if args.resume_pid == pid:
                        sid = args.resume_sid
                        seqs = seqs[sid:]
                    else:
                        sid = 0
                    pkg_spec = app.get_latent_specification(pkg)

                    if not args.print_sid and args.print_pid:
                        print('--resume-pid', pid, '--resume-sid', sid)
                    for s, seq in enumerate(seqs, sid):
                        sid = s
                        if args.print_sid:
                            print('--resume-pid', pid, '--resume-sid', sid)
                        for row in filter_anomalies(app, pkg_spec, seq):
                            print(json.dumps(row), file=fp)
            except KeyboardInterrupt:
                print('--resume-pid', pid, '--resume-sid', sid)

if __name__ == '__main__':
    main()

