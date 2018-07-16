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

def parse_ranges(expr:str) -> List[slice]:
    expr = expr.strip()
    if expr == '' or expr == '*':
        return [common.parse_slice(":")]
    return list(map(common.parse_slice, expr.split(",")))

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
