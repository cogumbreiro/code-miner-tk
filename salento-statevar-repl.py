#!/usr/bin/env python3
import sqlite3
import argparse
from tabulate import tabulate

import os.path
import sys
if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

from replui import handle_cursors
from cmd2 import argparse_completer

def parse_location(loc):
    pathname, lineno, *_ = loc.split(":", 3)
    PREFIX = "/media/usb1/revelant_src_files"
    if pathname.startswith(PREFIX):
        pathname = pathname[len(PREFIX):][len("2afd0766-9623-42a4-a81b-fa803db6d04f/") + 1:]
    return pathname, int(lineno)

STATE_TO_LABEL = {
    0:'condition',
    1:'function',
    2:'anywhere'
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
    state_var = Error(tbl.state_var.as_('kind')).as_('error')
    dirname = DirName(tbl.location).as_('directory')
    filename = BaseName(tbl.location).as_('file')
    path = Path(tbl.location).as_('path')
    lineno = Line(tbl.location).as_('line')
    # Other:
    total = fn.Count(tbl.star).as_('count')
    location = Loc(tbl.location).as_('location')

    def query(self):
        return Query.from_(self.tbl).where(self.tbl.symbol == '1')

    def as_str(self, field_or_fields):
        if hasattr(field_or_fields, "alias"):
            return field.alias
        else:
            return tuple(x.alias for x in field_or_fields)


Anomalies = Anomalies() # Singleton

def list_functions(cursor, limit=None, reverse=False, select_error=None, split_errors=False):
    headers = ["Function"]
    query = Anomalies.query()\
        .select(Anomalies.function)\
        .groupby(Anomalies.function)\
        .orderby(Anomalies.total, order=Order.asc if reverse else Order.desc)
    if split_errors:
        query = query.select(Anomalies.state_var).groupby(Anomalies.state_var)
        headers.append("Use result in")
    query = query.select(Anomalies.total, Anomalies.location)
    headers.append("Anomalies")
    headers.append("Example")
    if select_error is not None:
        query = query.where(Anomalies.state_var == select_error)

    if limit is not None:
        query = query.limit(limit)
    sql = query.get_sql()
    return cursor.execute(sql), headers

def list_dirs(cursor, sort_by=None, reverse=False):
    query = Anomalies.query().\
        select(Anomalies.dirname, Anomalies.total).\
        where(Anomalies.dirname != "").\
        groupby(Anomalies.dirname)
    if sort_by is not None:
        query = query.orderby(sort_by, order=Order.asc if reverse else Order.desc)
    return cursor.execute(query.get_sql()), ("Directory", "Anomalies")

def dir_exists(cursor, dirname):
    query = Anomalies.query().\
        select(Anomalies.dirname).\
        where(Anomalies.dirname != "" and Anomalies.dirname==dirname)
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

def file_exists(cursor, filename):
    query = Anomalies.query().\
        select(Anomalies.path).\
        where(Anomalies.path == filename)
    for row in cursor.execute(query.get_sql()):
        return True
    return False

def cat_file(cursor, filename):
    query = Anomalies.query().\
        select(Anomalies.lineno, Anomalies.function, Anomalies.state_var)\
        .where(Anomalies.path == filename)\
        .orderby(Anomalies.lineno, Anomalies.function)
    return cursor.execute(query.get_sql()), ('Line #', 'Function', 'Error: use result in')


import cmd2
from cmd2 import with_argparser

class REPL(cmd2.Cmd):
    prompt = '> '
    intro = 'Welcome to the Salento shell. Type help or ? to list commands.\nRun `dir` to list available directories. Run `funcs` to list anomalous functions.'
    cwd = None
    def __init__(self, get_cursor) -> None:
        self.settable.update({'cwd': 'The *logical* current working directory.'})
        history_file = '.' + os.path.splitext(__file__)[0] + ".hist"
        super().__init__(use_ipython=False, persistent_history_file=history_file)
        self.get_cursor = get_cursor
        self.allow_cli_args = False

    do_funcs = argparse.ArgumentParser()
    do_funcs.add_argument("--error", "-e",
        choices=tuple(STATE_TO_LABEL.values()),
        help="Only show the given error"
    )
    do_funcs.add_argument('--split', '-s', action='store_true')
    do_funcs.add_argument('--reverse', '-r', action='store_true')
    do_funcs.add_argument('--limit', '-l',
        type=int,
        default=30,
        help='Limit how many functions we list.'
    )
    @with_argparser(do_funcs)
    def do_funcs(self, args):
        with self.get_cursor() as cursor:
            elems, header = list_functions(cursor,
                limit=args.limit,
                select_error=args.error,
                split_errors=args.split,
                reverse=args.reverse,
            )
            self.ppaged(tabulate(elems, header))

    do_dirs = argparse.ArgumentParser()
    @with_argparser(do_dirs)
    def do_dirs(self, args):
        with self.get_cursor() as cursor:
            elems, headers = list_dirs(cursor, sort_by=Anomalies.total)
            self.ppaged(tabulate(elems, headers))

    do_dir = do_dirs

    def get_dirs(self):
        with self.get_cursor() as cursor:
            elems, headers = list_dirs(cursor)
            return (x for (x, y) in elems)

    do_cd = argparse.ArgumentParser()
    dir_act = do_cd.add_argument("dir", help="Change to the target directory.")
    setattr(dir_act, argparse_completer.ACTION_ARG_CHOICES, 'get_dirs')
    @with_argparser(do_cd)
    def do_cd(self, args):
        with self.get_cursor() as cursor:
            if dir_exists(cursor, args.dir):
                self.cwd = args.dir
                self.prompt = args.dir + " > "
            else:
                self.perror("Directory not found: {}".format(args.dir))
                self.perror("Run `dirs` first.")

    do_files = argparse.ArgumentParser()
    @with_argparser(do_files)
    def do_files(self, args):
        if self.cwd is None:
            self.perror("Run `dirs` first.")
            return
        with self.get_cursor() as cursor:
            elems, header = list_files(cursor, self.cwd, sort_by=Anomalies.total)
            self.ppaged(tabulate(elems, header))
    do_ls = do_files

    def get_files(self):
        if self.cwd is None:
            return ()
        with self.get_cursor() as cursor:
            elems, _ = list_files(cursor, self.cwd)
            return (x for x, y in elems)


    do_cat = argparse.ArgumentParser()
    cat_act = do_cat.add_argument("file", help="Show the errors of the current file.")
    setattr(cat_act, argparse_completer.ACTION_ARG_CHOICES, 'get_files')
    @with_argparser(do_cat)
    def do_cat(self, args):
        if self.cwd is None:
            self.perror("Run `dirs` first.")
            return
        filename = os.path.join(self.cwd, args.file)
        with self.get_cursor() as cursor:
            if not file_exists(cursor, filename):
                self.perror("Filename %r not found. Run `ls` first." % filename)
                return
            elems, header = cat_file(cursor, filename)
            self.ppaged(tabulate(elems, header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('db', help='input data file')
    args = parser.parse_args()
    with sqlite3.connect(args.db) as db:
        db.create_function("LOC", 1, lambda x: "{}:{}".format(*parse_location(x)))
        db.create_function("PATH", 1, lambda x: os.path.join("/", parse_location(x)[0]))
        db.create_function("BASENAME", 1, lambda x: os.path.basename(parse_location(x)[0]))
        db.create_function("DIRNAME", 1, lambda x: os.path.join("/", os.path.dirname(parse_location(x)[0])))
        db.create_function("LINE", 1, lambda x: parse_location(x)[1])
        db.create_function("ERROR", 1, lambda x:STATE_TO_LABEL[x])
        with handle_cursors(db) as get_cursor:
            repl = REPL(get_cursor)
            repl.cmdloop()

if __name__ == '__main__':
    main()