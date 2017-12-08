#!/usr/bin/env python3

import tarfile
import os
import sys
import os.path
import errno
import subprocess
import multiprocessing
import concurrent.futures

def delete_file(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def run(cmd, silent=True):
    kwargs = dict()
    fd = None
    try:
        if silent:
            fd = open(os.devnull, 'w')
            kwargs['stdout'] = fd
            kwargs['stderr'] = fd
        return subprocess.call(cmd, shell=True, **kwargs)
    finally:
        if fd is not None:
            fd.close()

def command(label, infile, outfile, cmd, show_command=False):
    if show_command:
        print(cmd)
    else:
        print(label + " " + outfile)
    try:
        if run(cmd) != 0:
            print("ERROR: " + cmd)
            delete_file(outfile)
            return False
    except:
        delete_file(outfile)
        raise
    return True

def target_filename(filename, prefix, extension):
    return os.path.join(prefix, filename + extension)

def processor(args, as2sal, ignore, tar, apisan):

    def do_run(tar_info):
        c_fname = tar_info.name
        as_fname = target_filename(c_fname, args.prefix, ".as")
        sal_fname = target_filename(c_fname, args.prefix, ".sal")
        sal_bz_fname = target_filename(c_fname, args.prefix, ".sal.bz2")
        o_file = os.path.splitext(os.path.basename(c_fname))[0] + ".o"
        if not c_fname.endswith(".c") or not tar_info.isfile():
            return
        if c_fname in ignore or as_fname in ignore or sal_fname in ignore or sal_bz_fname in ignore:
            print("SKIP " + c_fname)
            return
        if not os.path.exists(as_fname) and not os.path.exists(sal_bz_fname):
            # Extract file
            tar.extract(tar_info)
            # Cleanup
            tar.members = []
        def continuation(): # The rest should be scheduled in parallel
            if not os.path.exists(as_fname) and not os.path.exists(sal_bz_fname):
                # Compile file
                try:
                    if not command("APISAN", c_fname, as_fname, apisan + " compile " + c_fname, args.debug):
                        return
                finally:
                    # Remove filename
                    delete_file(c_fname)
                    delete_file(o_file)

            if not os.path.exists(sal_bz_fname):
                if command("SAN2SAL", as_fname, sal_bz_fname,
                        "python3 " + as2sal + " -i " + as_fname + " | bzip2 > " + sal_bz_fname, args.debug):
                    delete_file(as_fname)
            else:
                if args.debug:
                    print("# DONE " + sal_bz_fname)
                else:
                    print("DONE " + sal_bz_fname)
        return continuation
    return do_run

def par_run(executor, buffer_size, func, elems):
    def terminated(buff):
        # Try to find a completed future, otherwise, return the first item
        for (idx, x) in enumerate(buff):
            if x.done():
                yield idx

    def drain(buff, count):
        while len(buff) > count:
            fut = buff.pop(0)
            fut.result()
    buff = []
    try:
        for x in elems:
            ctl = func(x)
            # Check if there is a continuation
            if ctl is not None:
                buff.append(executor.submit(ctl))
            # Try to remove any terminated task
            for x in terminated(buff):
                del buff[x]
            # Otherwise, remove the first n tasks
            drain(buff, buffer_size)
        drain(buff, 0)
    except KeyboardInterrupt:
        # Cleanup pending commands
        for x in buff:
            x.cancel()
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Given an archive of C files, generates Salento Package JSON files.")
    parser.add_argument("-i", dest="infile", nargs='?', type=str,
                     default="/dev/stdin", help="A filename of the APISAN file format (.as). DEFAULT: '%(default)s'")
    parser.add_argument("-p", dest="prefix", nargs='?', type=str,
                     default="as-out", help="The directory where we are locating. DEFAULT: '%(default)s'")
    parser.add_argument("-d", help="Show commands instead of user friendly label.", dest="debug",
                    action="store_true")
    parser.add_argument("-s", dest="skip", nargs='+', default=[],
                     help="A list of files to ignore.")
    parser.add_argument("-t", dest="timeout", nargs='?', type=str,
                     default="1h", help="The timeout. DEFAULT: '%(default)s'")
    parser.add_argument("--nprocs", dest="nprocs", nargs='?', type=int,
                     default=multiprocessing.cpu_count(), help="The maximum number of parallel word counts. Default: %(default)s.")

    args = parser.parse_args()
    apisan = os.path.join(os.environ['APISAN_HOME'], 'apisan')
    if args.timeout is not None and args.timeout.strip() != "":
        apisan = "timeout " + args.timeout + " " + apisan
    tar = tarfile.open(args.infile, "r|*")
    as2sal = os.path.join(os.path.dirname(sys.argv[0]), 'apisan-to-salento.py')
    ignore = set(args.skip)
    do_run = processor(args, as2sal, ignore, tar, apisan)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.nprocs) as executor:
        par_run(executor=executor, buffer_size=args.nprocs, func=do_run, elems=tar)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # User break, it's ok to quit
        pass
