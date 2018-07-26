from typing import *

import functools
import argparse
import shlex
import string

# Local imports
import common

def argparse_cmd(fun):
    def wrapper(self, line):
        name = fun.__name__[3:]
        parser = argparse.ArgumentParser(description=fun.__doc__, prog=name)
        parser.exit = self.error
        getattr(self, 'argparse_' + name)(parser)
        try:
            try:
                args = shlex.split(line)
            except ValueError as e:
                raise REPLExit("Error parsing arguments of command %r: %s" % (name, e))
            fun(self, parser.parse_args(args))
        except REPLExit as e:
            print(e)
        except KeyboardInterrupt:
            pass
    wrapper.__name__ = fun.__name__
    wrapper.__doc__ = fun.__doc__
    return wrapper

class REPLExit(Exception):
    pass

class CallFormatter(string.Formatter):
    def format_field(self, value, spec):
        if spec == 'call':
            return value()
        else:
            return super(CallFormatter, self).format_field(value, spec)

def repl_format(*args, **kwargs) -> str:
    fmt = CallFormatter()
    try:
        return fmt.format(*args, **kwargs)
    except (TypeError, KeyError, ValueError, AttributeError) as e:
        raise REPLExit("Error parsing format: %s" % e)

from contextlib import contextmanager

class ResourceHandler:
    def __init__(self, db):
        self.db = db
        self.cursors = []

    @contextmanager
    def manage_cursors(self):
        yield self
        for x in self.cursors:
            x.close()
        self.cursors.clear()

    @contextmanager
    def get_cursor(self):
        if len(self.cursors) == 0:
            cursor = self.db.cursor()
        else:
            cursor = self.cursors.pop()
        yield cursor
        self.cursors.append(cursor)

@contextmanager
def handle_cursors(db):
    handler = ResourceHandler(db)
    with handler.manage_cursors():
        yield handler.get_cursor
