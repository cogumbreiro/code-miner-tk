#!/usr/bin/env python3
import sqlite3
import argparse
import shlex
from tabulate import tabulate

import os.path
import sys
if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

from replui import handle_cursors
from cmd2 import argparse_completer

def parse_location(loc, skip_hash=True):
    pathname, loc = loc.split(":", 1)
    lineno = loc.split(':', 1)[0]
    #pathname, lineno, *_ = loc.split(":", 3)
    PREFIX = "/media/usb1/revelant_src_files/"
    if pathname.startswith(PREFIX):
        pathname = pathname[len(PREFIX):]
        if skip_hash:
            pathname = pathname[len("2afd0766-9623-42a4-a81b-fa803db6d04f/"):]
    return pathname, int(lineno)

STATE_TO_LABEL = {
    0:'condition',
    1:'argument',
    2:'anywhere'
}

STATE_SYM_TO_STR = {
    (0, '0'):'branch',
    (0, '1'):'not branch',
    (1, '0'):'pass to func',
    (1, '1'):'not pass to func',
    (2, '0'):'use result',
    (2, '1'):'ignore result'
}


import os.path
from pypika import Query, Table, Field, Order
from pypika import functions as fn

class DirName(fn.Function):
    def __init__(self, term, alias=None):
        super().__init__('DIRNAME', term, alias=alias)
class Path(fn.Function):
    def __init__(self, term, alias=None):
        super().__init__('PATH', term, alias=alias)
class OsPath(fn.Function):
    def __init__(self, term, alias=None):
        super().__init__('OS_PATH', term, alias=alias)
class BaseName(fn.Function):
    def __init__(self, term, alias=None):
        super().__init__('BASENAME', term, alias=alias)
class Loc(fn.Function):
    def __init__(self, term, alias=None):
        super().__init__('LOC', term, alias=alias)
class Line(fn.Function):
    def __init__(self, term, alias=None):
        super().__init__('LINE', term, alias=alias)
class Error(fn.Function):
    def __init__(self, term, alias=None):
        super().__init__('ERROR', term, alias=alias)


class Anomalies:
    tbl = Table('anomalies')
    function = tbl.call.as_('function')
    symbol = tbl.symbol.as_('symbol')
    state_var = Error(tbl.state_var.as_('kind')).as_('error')
    dirname = DirName(tbl.location).as_('directory')
    filename = BaseName(tbl.location).as_('file')
    path = Path(tbl.location).as_('path')
    os_path = OsPath(tbl.location).as_('os_path')
    lineno = Line(tbl.location).as_('line')
    # Other:
    total = fn.Count(tbl.star).as_('count')
    location = Loc(tbl.location).as_('location')

    def query(self):
        return Query.from_(self.tbl)

    def as_str(self, field_or_fields):
        if hasattr(field_or_fields, "alias"):
            return field.alias
        else:
            return tuple(x.alias for x in field_or_fields)


def get_confidence(cursor, func_name, state_var, sym):
    for (score,) in cursor.execute(
        "SELECT score FROM state_confidence WHERE call=? AND state_var=? AND symbol=?",
        (func_name, state_var, sym)):
        return score
    return 0.0

Anomalies = Anomalies() # Singleton

def list_functions(cursor, cursor2, limit=None, reverse=False,
        select_error=None, split_errors=False, only_func=None, select_sym=None, show_all=False):
    headers = ["Function"]
    query = Anomalies.query()\
        .select(Anomalies.function)\
        .groupby(Anomalies.function)\
        .orderby(Anomalies.total, order=Order.asc if reverse else Order.desc)

    if only_func is not None:
        query = query.where(Anomalies.function == only_func)

    if split_errors:
        query = query \
            .select(Anomalies.tbl.state_var,Anomalies.symbol) \
            .groupby(Anomalies.tbl.state_var,Anomalies.symbol)
        headers.append("Result should")
        headers.append("Confidence")
    query = query.select(Anomalies.total, Anomalies.location)
    headers.append("Anomalies")
    headers.append("Example")
    if select_sym is not None:
        query = query.where(Anomalies.symbol == select_sym)
    if select_error is not None:
        query = query.where(Anomalies.state_var == select_error)

    if limit is not None:
        query = query.limit(limit)
    sql = query.get_sql()
    elems = cursor.execute(sql)
    if split_errors:
        elems = (
            (call, STATE_SYM_TO_STR[(state_var,sym)], "{:.0%}".format(get_confidence(cursor2, call, state_var, sym)), anom, ex)
            for (call,state_var,sym,anom,ex) in elems
        )

    if only_func is not None:
        elems = (row[1:] for row in elems)
        headers = headers[1:]

    return elems, headers

def list_func_anomalies(cursor, func_name, state_var, symbol):
    query = Anomalies.query() \
        .select(Anomalies.location) \
        .where(Anomalies.function == func_name) \
        .where(Anomalies.state_var == state_var) \
        .where(Anomalies.symbol == symbol) \
        .orderby(Anomalies.path, Anomalies.lineno) \
        .distinct()
    return cursor.execute(query.get_sql()), ["Location"]

def list_dirs(cursor, sort_by=None, reverse=False, limit=None):
    query = Anomalies.query().\
        select(Anomalies.dirname, Anomalies.total).\
        groupby(Anomalies.dirname)
    if limit is not None:
        query = query.limit(limit)
    if sort_by is not None:
        query = query.orderby(sort_by, order=Order.asc if reverse else Order.desc)
    return cursor.execute(query.get_sql()), ("Directory", "Anomalies")

def dir_exists(cursor, dirname):
    query = Anomalies.query().\
        select(Anomalies.dirname).\
        where(Anomalies.dirname==dirname)
    for row in cursor.execute(query.get_sql()):
        return True
    return False

def list_files(cursor, dirname, sort_by=None, reverse=False):
    query = Anomalies.query().\
        select(Anomalies.filename, Anomalies.total).\
        where(Anomalies.dirname == dirname).\
        groupby(Anomalies.filename)
    if sort_by is not None:
        query = query.orderby(sort_by, order=Order.asc if reverse else Order.desc)
    return (cursor.execute(query.get_sql()), ("File", "Anomalies"))

def get_all_files(cursor):
    query = Anomalies.query().\
        select(Anomalies.os_path).\
        orderby(Anomalies.os_path).\
        groupby(Anomalies.os_path)
    return (x for (x,) in cursor.execute(query.get_sql()))

def path_to_os_path(cursor, filename):
    query = Anomalies.query().\
        select(Anomalies.os_path).\
        where(Anomalies.path == filename)
    for (fname,) in cursor.execute(query.get_sql()):
        return fname


def file_exists(cursor, filename):
    query = Anomalies.query().\
        select(Anomalies.path).\
        where(Anomalies.path == filename)
    for row in cursor.execute(query.get_sql()):
        return True
    return False

def cat_file(cursor, cursor2, filename):
    query = Anomalies.query().\
        select(Anomalies.lineno, Anomalies.function, Anomalies.tbl.state_var, Anomalies.symbol)\
        .where(Anomalies.path == filename)\
        .orderby(Anomalies.lineno, Anomalies.function)
    elems = cursor.execute(query.get_sql())
    elems = (
        (
            lineno,
            call,
            STATE_SYM_TO_STR[(sv,sym)],
            "{:.0%}".format(get_confidence(cursor2, call, sv, sym))
        )
        for (lineno, call, sv, sym) in elems
    )
    return elems, ('Line #', 'Call', 'Result should', 'Confidence')


import cmd2
from cmd2 import with_argparser

class REPL(cmd2.Cmd):
    prompt = '> '
    intro = 'Welcome to the Salento shell. Type help or ? to list commands.\nRun `dir` to list available directories. Run `funcs` to list anomalous functions.'
    cwd = None
    def __init__(self, get_cursor, history_file) -> None:
        super().__init__(use_ipython=False, persistent_history_file=history_file)
        self.get_cursor = get_cursor
        self.allow_cli_args = False

    do_funcs = argparse.ArgumentParser()
    do_funcs.add_argument("--error", "-e",
        choices=tuple(STATE_TO_LABEL.values()),
        help="Only show the given error type"
    )
    do_funcs.add_argument('--group', '-g',
        dest='split',
        action='store_false',
        help="Group all error types togethers."
    )
    do_funcs.add_argument('--reverse', '-r', action='store_true')
    do_funcs.add_argument('--limit', '-l',
        type=int,
        help='Limit how many functions we list.'
    )
    do_funcs.add_argument('--func', '-f',
        help="Only show results for the given function"
    )
    do_funcs.add_argument('--should',
        dest="select_sym",
        action="store_const",
        const="0",
        help="Only show SHOULD recommendations"
    )
    do_funcs.add_argument('--shouldnot',
        dest="select_sym",
        action="store_const",
        const="1",
        help="Only show SHOULD NOT recommendations"
    )
    @with_argparser(do_funcs)
    def do_funcs(self, args):
        """Lists all anomalies, grouping the reports by function name."""
        with self.get_cursor() as cursor, self.get_cursor() as cursor2:
            elems, header = list_functions(cursor,
                cursor2,
                limit=args.limit,
                select_error=args.error,
                split_errors=args.split,
                reverse=args.reverse,
                only_func=args.func,
                select_sym=args.select_sym,
            )
            self.ppaged(tabulate(elems, header))

    do_anom = argparse.ArgumentParser()
    do_anom.add_argument("func", help="Function name.")
    do_anom.add_argument("state",
        choices=tuple(STATE_TO_LABEL.values()),
        help="Only show the given error type"
    )
    do_anom.add_argument("symbol",
        choices=("should", "shouldnot"),
        help="Recommendation"
    )
    @with_argparser(do_anom)
    def do_anom(self, args):
        with self.get_cursor() as cursor:
            elems, headers = list_func_anomalies(cursor,
                args.func,
                args.state,
                '0' if args.symbol == "should" else '1'
            )
            self.ppaged(tabulate(elems, headers))

    do_dirs = argparse.ArgumentParser()
    do_dirs.add_argument('--limit', '-l',
        type=int,
        help='Limit how many functions we list.'
    )
    @with_argparser(do_dirs)
    def do_dirs(self, args):
        """Lists all directories in the anomaly database."""
        with self.get_cursor() as cursor:
            elems, headers = list_dirs(cursor,
                sort_by=Anomalies.total,
                limit=args.limit
            )
            self.ppaged(tabulate(elems, headers))

    def get_dirs(self):
        with self.get_cursor() as cursor:
            elems, headers = list_dirs(cursor)
            return (x for (x, y) in elems)

    do_chdir = argparse.ArgumentParser()
    setattr(do_chdir.add_argument("dir", help="Change to the target directory."),
        argparse_completer.ACTION_ARG_CHOICES, 'get_dirs'
    )
    @with_argparser(do_chdir)
    def do_chdir(self, args):
        """Sets the current working directory. See command 'files'."""
        with self.get_cursor() as cursor:
            if dir_exists(cursor, args.dir):
                self.cwd = args.dir
                self.prompt = args.dir + " > "
            else:
                self.pfeedback("Directory {!r} was not found.".format(args.dir))
                self.pfeedback("Run 'dirs' to list existing directories.")

    do_files = argparse.ArgumentParser()
    setattr(do_files.add_argument("dir", nargs="?", help="Supply a directory to list."),
        argparse_completer.ACTION_ARG_CHOICES, 'get_dirs'
    )
    @with_argparser(do_files)
    def do_files(self, args):
        if args.dir is None and self.cwd is None:
            self.pfeedback("Run 'chdir' first.")
            return
        dirname = self.cwd if args.dir is None else args.dir
        with self.get_cursor() as cursor:
            elems, header = list_files(cursor, dirname, sort_by=Anomalies.total)
            self.ppaged(tabulate(elems, header))

    def get_files(self):
        if self.cwd is None:
            return ()
        with self.get_cursor() as cursor:
            elems, _ = list_files(cursor, self.cwd)
            return (x for x, y in elems)


    do_show = argparse.ArgumentParser()
    setattr(
        do_show.add_argument("file", help="Show the errors of the current file."),
        argparse_completer.ACTION_ARG_CHOICES, 'get_files'
    )
    do_show.add_argument('--line', '-l', type=int, help="Show the source code instead of the anomalies.")
    @with_argparser(do_show)
    def do_show(self, args):
        if not args.file.startswith("/") and self.cwd is None:
            self.pfeedback("Run 'chdir' first.")
            return

        if ":" in args.file:
            filename, lineno = args.file.split(":", 1)
            lineno = int(lineno)
        else:
            filename = args.file
            lineno = None

        if args.line is not None:
            lineno = args.line

        if not filename.startswith("/"):
            filename = os.path.join(self.cwd, filename)

        if lineno is not None:
            with self.get_cursor() as cursor:
                filename = path_to_os_path(cursor, filename)
                if filename is None:
                    self.pfeedback("Filename %r not found.\nRun 'files' first." % filename)
                    return

                hl_line = """{if (NR == %d){print "\033[07m" $0 "\033[27m"} else {print $0}}""" % lineno
                hl_line = "awk %s %s" % (shlex.quote(hl_line), shlex.quote(filename))
                os.system('%s | less -R -N +%d -j 10' % (hl_line, lineno))
            return

        with self.get_cursor() as cursor, self.get_cursor() as cursor2:
            if not file_exists(cursor, filename):
                self.pfeedback("Filename %r not found.\nRun 'files' first." % filename)
                return
            elems, header = cat_file(cursor, cursor2, filename)
            self.ppaged(tabulate(elems, header))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db', help='input data file')
    curr_file = os.path.basename(__file__)
    hist_file = "." + os.path.splitext(curr_file)[0] + ".hist"
    parser.add_argument('-l',
        dest="history_file",
        default=hist_file,
        help="Intepreter history log. Default: %(default)r."
    )
    parser.add_argument('--prefix', '-p',
        default=".",
        help="The prefix path where the source code is located."
    )
    parser.add_argument('--print-files', action='store_true', help="Prints all files in the anomalies.")

    args = parser.parse_args()
    def os_path(path):
        path = parse_location(path, skip_hash=False)[0]
        return os.path.join(args.prefix, path)
        #return args.prefix + path # if path.startswith('/') else os.path.join(args.prefix, path)
    with sqlite3.connect(args.db) as db:
        db.create_function("LOC", 1, lambda x: "/{}:{}".format(*parse_location(x)))
        db.create_function("PATH", 1, lambda x: os.path.join("/", parse_location(x)[0]))
        db.create_function("OS_PATH", 1, os_path)
        db.create_function("BASENAME", 1, lambda x: os.path.basename(parse_location(x)[0]))
        db.create_function("DIRNAME", 1, lambda x: os.path.join("/", os.path.dirname(parse_location(x)[0])))
        db.create_function("LINE", 1, lambda x: parse_location(x)[1])
        db.create_function("ERROR", 1, lambda x:STATE_TO_LABEL[x])
        with handle_cursors(db) as get_cursor:
            if args.print_files:
                with get_cursor() as cursor:
                    for x in get_all_files(cursor):
                        print(x)
                return
            repl = REPL(get_cursor, args.history_file)
            repl.cmdloop()

if __name__ == '__main__':
    main()