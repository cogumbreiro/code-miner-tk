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

# Shut up Tensorflow
if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

from argparse import Namespace
from common import cons_last, memoize
from replui import argparse_cmd, parse_ranges, repl_format, REPLExit
from salui import make_app, ASequence
from collections import Counter
from statedist import CallDistNorm, TermDistNorm

import enforce

USED_IN_COND = 0
USED_IN_ARG = 1
USED_ANY = 2

STATE_TO_LABEL = {
    USED_IN_COND:'be checked',
    USED_IN_ARG:'be used in arg',
    USED_ANY:'be used'
}

def parse_location(loc:str) -> Tuple[str,int]:
    pathname, lineno, *_ = loc.split(":", 3)
    PREFIX = "/media/usb1/revelant_src_files/"
    if pathname.startswith(PREFIX):
        pathname = pathname[len(PREFIX):]
    return pathname, int(lineno)

SKIP_IF_VALUE = {
    USED_IN_COND: {
        True: set([
            "g_build_filename",
            "g_variant_new_strv",
            "g_strjoin",
            "g_malloc0",
            "g_strdup_printf",
            "g_regex_replace_literal",
            "g_hash_table_new" ,
        ]),
        False: set([]),
    },
    USED_IN_ARG: {
        True: set([]),
        False: set([]),
    },
    USED_ANY: {
        True: set([]),
        False: set([]),
    }
}

SKIP_CHECK = {USED_IN_COND:
    set([
        "g_ascii_digit_value", # If c is a decimal digit (according to g_ascii_isdigit()), its numeric value. Otherwise, -1.
        "g_date_get_julian",
        "g_dgettext",
        "g_file_get_contents",
        "g_get_home_dir", # (const gchar *) the current user's home directory.
        "g_get_user_config_dir",
        "g_getenv", # const char *
        "g_key_file_get_integer",
        "g_key_file_get_integer_list",
        "g_key_file_get_boolean", # gboolean
        "g_key_file_get_string_list",
        "g_mkdir_with_parents",
        "g_node_insert_before", # the inserted GNode
        "g_list_length",
        "g_quark_from_static_string",
        "g_quark_from_string", # the GQuark identifying the string, or 0 if string is NULL
        "g_random_int",
        "g_regex_match_full", # the example does not test the return value
        "g_spawn_command_line_sync",
        "g_spawn_sync", # TRUE on success, FALSE if an error was set
        "g_slist_length",
        "g_slist_reverse",
        "g_slist_position",
        "g_strchug",
        "g_strchomp",
        "g_strerror", # const gchar *
        "g_string_free",
        "g_timeout_add",
        "g_timeout_add_full",
        "g_unichar_to_utf8",
        "g_unlink", # int
        "g_utf8_get_char",
        "g_utf8_offset_to_pointer",
        "g_utf8_strlen",
        "g_variant_get_child_value",
        "g_variant_iter_n_children", # the number of children in the container
        "g_variant_n_children", # the number of children in the container
    ]),
    USED_IN_ARG: set([
        # Conservative
        "g_ascii_strtod",
        "g_getenv",
        "g_hash_table_lookup",
        "g_hash_table_new",
        "g_hash_table_new_full",
        "g_malloc_n",
        "g_malloc0_n",
        "g_match_info_fetch",
        "g_key_file_get_boolean",
        "g_key_file_get_string",
        "g_key_file_get_integer",
        "g_key_file_get_integer_list",
        "g_list_length",
        "g_quark_from_string",
        "g_random_int",
        "g_slist_concat",
        "g_slist_length",
        "g_slist_reverse",
        "g_string_append",
        "g_string_insert_len",
        "g_strdup",
        "g_strdup_printf",
        "g_str_equal",
        "g_string_append",
        "g_string_insert_len",
        "g_strv_length",
        "g_utf8_find_prev_char",
        "g_utf8_get_char",
        "g_utf8_offset_to_pointer",
        "g_variant_iter_n_children", # the number of children in the container
        "g_variant_n_children", # the number of children in the container
    ]),
    USED_ANY: set([
        "g_io_add_watch",
        "g_ptr_array_free",
        "g_string_free",
        "g_spawn_sync",
        "g_spawn_command_line_sync",
        "g_mkstemp",
        "g_timeout_add",
        "g_timeout_add_seconds",
        "g_timeout_add_seconds_full",
        "g_idle_add",
        "g_io_add_watch_full",
        "g_thread_new",
        "g_test_run",
        "g_datalist_id_remove_no_notify",
        "g_variant_ref_sink",
    ]),
}
TRESHOLD = 0.2

CBGetStates = Callable[[CallDistNorm],Iterable[Tuple[int,TermDistNorm]]]
T = TypeVar('T')
CBGetIds = Callable[[CallDistNorm],T]

def get_anomalous_states(call:CallDistNorm, treshold:float=TRESHOLD) -> Iterable[Tuple[int,TermDistNorm]]:
    for idx, st in enumerate(call.states):
        if st.normalized_prob < treshold and st.get_max()[1] >= treshold:
            yield (idx, st)

class AnomalyEngine:
    def get_anomalous_states(self, call:CallDistNorm) -> Iterable[Tuple[int,TermDistNorm,float]]:
        pass

    def score_message(self, st:TermDistNorm) -> str:
        pass

    def get_call_score(self, call:CallDistNorm) -> float:
        pass

    def get_sequence_score(self, seq:ASequence) -> float:
        pass

    def count(self, seq:ASequence) -> int:
        count = 0
        for call in seq.get_state_probs():
            for _ in self.get_anomalous_states(call):
                count += 1
        return count


class FilterSequenceAnomaly2(AnomalyEngine):
    def __init__(self, filter_idx:Optional[int]=None, treshold:float=TRESHOLD) -> None:
        self.filter_idx = filter_idx
        self.treshold = treshold
        self.freqs = {0: Counter(), 1:Counter(), 2:Counter()}
        self.visited = set()
        self.visited_seqs = set()
        self.max_freqs = [0, 0, 0]

    def get_anomalous_ids(self, call:CallDistNorm) -> Iterator[Tuple[str,int]]:
        for idx, _ in get_anomalous_states(call, treshold=self.treshold):
            if self.filter_idx is None or self.filter_idx == idx:
                yield idx

    def get_anomalous_states(self, call:CallDistNorm) -> Iterable[Tuple[int,TermDistNorm,float]]:
        for idx, st in get_anomalous_states(call, treshold=self.treshold):
            if self.filter_idx is not None and self.filter_idx != idx:
                continue
            # If the return value is used somehow, then that's not anomalous
            state = parse_value(st.name)
            if state:
                continue
            score = self.compute_score(call, idx, st)
            if score > 0:
                yield (idx, st, score)

    def get_freq(self, call, idx):
        return self.freqs[idx].get(call.name, 0)

    def inc_freq(self, call, idx):
        self.freqs[idx][call.name] += 1

    def get_dampen(self, call, idx):
        freq = self.get_freq(call, idx)
        mf = max(freq, self.max_freqs[idx])
        self.max_freqs[idx] = mf
        return 1 - freq/mf

    def compute_score(self, call, idx, st):
        dampen = self.get_dampen(call, idx)
        score = (1 - st.normalized_prob) * dampen
        return score

    def score_message(self, call, idx, st) -> str:
        return "(score: {:.1f}; freq: {}/{}) ".format(self.compute_score(call, idx, st), self.get_freq(call, idx), self.max_freqs[idx])

    def get_call_score(self, call:CallDistNorm) -> float:
        return sum(map(itemgetter(2), self.get_anomalous_states(call)))

    def on_sequence(self, pkg, seq):
        uid = (pkg.pid, seq.sid)
        if uid in self.visited_seqs:
            return
        for evt, call in zip(seq, seq.get_state_probs()):
            for idx in self.get_anomalous_ids(call):
                cid = (evt.location, call.name, idx)
                if cid in self.visited:
                    continue
                self.visited.add(cid)
                self.inc_freq(call, idx)
        self.visited_seqs.add(uid)

    def load_sequences(self, seqs):
        result = []
        for pkg, seq in seqs:
            self.on_sequence(pkg, seq)
            result.append((pkg, seq))
        for idx, counter in self.freqs.items():
            if len(counter) == 0:
                continue
            (_, mf), = counter.most_common(1)
            self.max_freqs[idx] = mf
        return result


    def get_sequence_score(self, seq:ASequence) -> float:
        return sum(self.get_call_score(call) for call in seq.get_state_probs())

class FilterSequenceAnomaly(AnomalyEngine):
    def __init__(self, filter_idx:Optional[int]=None, treshold:float=TRESHOLD) -> None:
        self.filter_idx = filter_idx
        self.treshold = treshold

    def get_anomalous_states(self, call:CallDistNorm) -> Iterable[Tuple[int,TermDistNorm,float]]:
        for idx, st in get_anomalous_states(call, treshold=self.treshold):
            if call.name in SKIP_CHECK[idx]:
                continue
            # If the return value is used somehow, then that's not anomalous
            state = parse_value(st.name)
            if state:
                continue
            if self.filter_idx is None or idx == self.filter_idx:
                score = 1 - st.normalized_prob
                yield (idx, st, score)

    def load_sequences(self, seqs):
        return seqs

    def score_message(self, call, idx, st:TermDistNorm) -> str:
        return "(+{:.0%}) ".format(st.get_max()[1] - st.prob)

    def get_call_score(self, call:CallDistNorm) -> float:
        return sum(map(itemgetter(2), self.get_anomalous_states(call)))

    def get_sequence_score(self, seq:ASequence) -> float:
        return sum(self.get_call_score(call) for call in seq.get_state_probs())


def show_seq(seq:ASequence, anomaly_engine:AnomalyEngine, only_show_anomalous:bool=True) -> None:
    node = "{}".format
    is_first = True
    pathname = None
    lineno = None
    print("Anomaly found:", "{:.0f}".format(anomaly_engine.count(seq)))
    for event, call in zip(seq, seq.get_state_probs()):
        highest_key, highest = call.get_max()
        label = call.name
        loc = event.location
        new_pathname, lineno = parse_location(loc)
        if new_pathname != pathname:
            pathname = new_pathname
            print("Source:", pathname)
        label = str(lineno) + ":" + label

        row = []
        anomalies = anomaly_engine.get_anomalous_states(call)
        for idx, st, score in anomalies:
            recommendation = "should NOT" if st.name == "1" else "SHOULD"
            recommendation += " " + STATE_TO_LABEL[idx]
            data = {
                'index':idx,
                'state':st,
                'score_msg': anomaly_engine.score_message(call, idx, st),
                'recommendation': recommendation,
                'score': score,
            }
            msg = "{score_msg}return value {recommendation}\t"
            row.append(msg.format(**data))
        if not only_show_anomalous or only_show_anomalous and len(row) > 0:
            row.insert(0, label)
        else:
            continue


        print("\t".join(row))
        is_first = False

    print(". " * 40)
    print()

def parse_value(value:str) -> bool:
    return True if value == "1" else False

def get_location(seq:ASequence) -> str:
    return parse_location(seq.last_location)[0]

def flatten_sequences(pkgs:Iterable[ASequence], seq_ids:List[slice]) -> Iterable[ASequence]:
    for pkg in pkgs:
        seqs:Iterable[ASequence] = pkg
        if seq_ids is not None:
            accept = set()
            for sids in seq_ids:
                for elem in range(*sids.indices(len(pkg))):
                    accept.add(elem)
            seqs = filter(lambda x: x.sid in accept, seqs)
        yield from ((pkg, seq) for seq in seqs)


class REPL(cmd.Cmd):
    prompt = '> '
    intro = 'Welcome to the Salento shell. Type help or ? to list commands.\n'
    def __init__(self, app:Any) -> None:
        cmd.Cmd.__init__(self)
        self.app = app
        self.ae:AnomalyEngine = FilterSequenceAnomaly2()

    def do_pkgs(self, line:str) -> None:
        """
        Lists all packages.
        """
        for pkg in self.app.pkgs:
            print(pkg.name)

    def error(self, error_code:int=2, msg:Optional[str]=None) -> None:
        if msg is not None:
            print(msg, file=sys.stderr)
        raise REPLExit

    FORMAT = 'pid: {pkg.pid} sid: {seq.sid} count: {seq.count} last: {last_location} anomaly score: {score:.1f}'

    def argparse_seq(self, parser:Any) -> None:
        # Filter which packages.
        parser.add_argument('--pid', default='*', help="A query to match packages, the format is a Python slice expression, so ':' retreives all packages in the dataset. You can also use '*' to match all elements.")
        parser.add_argument('--sid', default='*', help="A query to select sequences, by default we match all ids. You can use '*' to match all sequences.")
        # Save visualization
        parser.add_argument('--print', action='store_true', help='Visualize the trace on the screen.')
        # Sort the final list
        parser.add_argument('--gt', type=float, default=0.01, help='Filter any value below this treshold.')
        parser.add_argument('--only', type=int, help='Only show the given type of anomaly.')

    @argparse_cmd
    def do_seq(self, args:Namespace) -> None:
        """
        Run queries at the sequence level.
        """

        app = self.app
        try:
            pkg_ids = parse_ranges(args.pid)
            seq_ids = parse_ranges(args.sid) if args.sid is not None else None
        except ValueError as e:
            raise REPLExit("Error parsing pkg-ids %r:" % args.pid, str(e))

        fmt = self.FORMAT
        self.ae.filter_idx=args.only
        for pkg, seq in self.ae.load_sequences(flatten_sequences(app.pkgs.lookup(pkg_ids), seq_ids)):

            score = self.ae.get_sequence_score(seq)
            if score < args.gt:
                continue

            def do_fmt(x:str) -> str:
                return repl_format(x, pkg=pkg, seq=seq, score=score, last_location=get_location(seq))

            if args.print:
                show_seq(seq, self.ae)
            else:
                print(do_fmt(fmt))



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input data file')
    parser.add_argument('--dirname', '-d', default="save",
                        help='directory to load model from')
    args = parser.parse_args()

    with make_app(args.filename, args.dirname) as aggregator:
        aggregator.init()
        repl = REPL(aggregator)
        repl.cmdloop()

if __name__ == '__main__':
    main()
