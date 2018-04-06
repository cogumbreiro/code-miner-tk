#!/usr/bin/env python3
import os.path
import string
import time
import subprocess
from functools import reduce

def resolve_path(dirname, fname):
    if os.path.isabs(fname):
        return fname
    return os.path.join(dirname, fname)

def depth_first_search(elems, get_outgoing):
    visited = set()
    to_process = list(elems)
    while len(to_process) > 0:
        elem = to_process.pop()
        if elem in visited:
            continue
        yield elem
        visited.add(elem)
        to_process.extend(set(get_outgoing(elem)) - visited)

def topological_sort(elems, get_outgoing):
    data = {x:set(get_outgoing(x)) for x in elems}
    for k, v in data.items():
        v.discard(k) # Ignore self dependencies
    while True:
        ordered = set(item for item,dep in data.items() if not dep)
        if len(ordered) == 0:
            break
        yield from ordered
        data = {item: (dep - ordered) for item,dep in data.items()
                if item not in ordered}
    if len(data) > 0:
        raise ValueError("A cyclic dependency exists amongst %r" % data)

def make_sort(elems, get_outgoing):
    elems = depth_first_search(elems, get_outgoing)
    for elem in topological_sort(elems, get_outgoing):
        yield elem

class Rule:
    def __init__(self, sources, targets, fun):
        self.targets = targets
        self.sources = sources
        self.fun = fun
        self.name = fun.__name__

    def get_missing_targets(self, ctx):
        return (x for x in self.targets if ctx.get_time(x) is None)

    def needs_update(self, ctx):
        sources = None
        for (t_name, t_time) in map(lambda x:(x, ctx.get_time(x)), self.targets):
            if t_time is None:
                return True
            # Cache getting the time of the sources
            if sources is None:
                sources = list(map(lambda x:(x, ctx.get_time(x)), self.sources))
            for (s_name, s_time) in sources:
                if ctx.target_needs_update(s_time, t_time):
                    return True
        return False

    def run(self, ctx, *args, **kwargs):
        if self.needs_update(ctx):
            self.fun(ctx, *args, **kwargs)
            for x in self.get_missing_targets(ctx):
                raise ValueError("Rule %r did not create target %r" % (self.name, ctx.get_path(x)))

    def __repr__(self):
        return "Rule(name=%r, sources=%r, targets=%r)" % (self.name, self.sources, self.targets)

class Rules:
    def __init__(self, get_path, elems, build_missing_sources=True):
        self.get_path = get_path
        self.rules = {}
        elems = list(elems)
        for rule in elems:
            for target in rule.targets:
                path = get_path(target)
                if path in self.rules:
                    raise ValueError("Rule %r and rule %r generate the same target target %r, file %r" % (self.rules[path].name, rule.name, target, path))
                self.rules[path] = rule
        if build_missing_sources:
            for rule in elems:
                for src in rule.sources:
                    path = get_path(src)
                    if path not in self.rules:
                        rule = lambda *args, **kwargs: None
                        rule.__name__ = src
                        self.rules[path] = Rule([], [src], rule)



    def get_rule(self, target):
        try:
            path = self.get_path(target)
            return self.rules[path]
        except KeyError as err:
            raise ValueError("Target %r (resolved as %r) does not exist and there is no rule to create it." % (target, path))

    def get_rules(self, targets):
        return map(self.get_rule, targets)

    def foreach_target(self, targets):
        return self.foreach_rule(self.get_rules(targets))

    def foreach_rule(self, rules):
        return make_sort(rules, lambda x: self.get_rules(x.sources))

class Makefile:
    def __init__(self):
        self.rules = []

    def rule(self, target=None, targets=(), source=None, sources=()):
        if target is not None:
            targets = list(targets)
            targets.append(target)
        if source is not None:
            sources = list(sources)
            sources.append(source)
        rules = self.rules
        def wrapper(fun):
            rule = Rule(sources, targets, fun)
            rules.append(rule)
            return rule

        return wrapper

    def make(self, ctx, *args, **kwargs):
        if "targets" in kwargs:
            targets = list(kwargs["targets"])
            del kwargs["targets"]
        else:
            targets = []

        if "target" in kwargs:
            targets.append(kwargs["target"])
            del kwargs["target"]

        rules = Rules(ctx.get_path, self.rules)
        for rule in rules.foreach_target(targets):
            rule.run(ctx, *args, **kwargs)

    def run(self, ctx, rules, *args, **kwargs):
        db = Rules(ctx.get_path, self.rules)
        for rule in db.foreach_rule(rules):
            rule.run(ctx, *args, **kwargs)

class EnvResolver:
    def __init__(self, dirname, env):
        self.dirname = dirname
        self.env = env

    def __call__(self, filename):
        fmt = string.Formatter()
        filename = fmt.vformat(filename, (), self.env)
        return os.path.abspath(resolve_path(self.dirname, filename))

class FileCtx:
    def __init__(self, get_path=os.path.abspath):
        self.get_path = get_path
    
    def get_time(self, path):
        try:
            return os.path.getmtime(self.get_path(path))
        except FileNotFoundError:
            return None

    def target_needs_update(self, source, target):
        if target is None:
            return True
        if source is None:
            raise ValueError("Error source is null!")
        return target < source

