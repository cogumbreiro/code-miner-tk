#!/usr/bin/env python3

import tarfile
import os
import sys
import os.path

def target_filename(filename, prefix, extension):
    return os.path.join(prefix, filename + extension)

def do_check(filename, prefix):
    filename2 = filename.split(os.path.sep, 1)
    if len(filename2) == 1:
        return check_filename(filename, prefix)
    else:
        return check_filename(filename, prefix) or check_filename(filename2[1], prefix)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Given an archive of C files, generates Salento Package JSON files.")
    parser.add_argument("-i", dest="infile", nargs='?', type=str,
                     default="/dev/stdin", help="A filename of the APISAN file format (.as). DEFAULT: '%(default)s'")
    parser.add_argument("-p", dest="prefix", nargs='?', type=str,
                     default="as-out", help="The directory where we are locating. DEFAULT: '%(default)s'" )

    args = parser.parse_args()
    apisan = os.path.join(os.environ['APISAN_HOME'], 'apisan')
    tar = tarfile.open(args.infile, "r|*")
    as2sal = os.path.join(os.path.dirname(sys.argv[0]), 'apisan-to-salento.py')
    for tar_info in tar:
        c_fname = tar_info.name
        if not c_fname.endswith(".c"):
            continue
        as_fname = target_filename(c_fname, args.prefix, ".as")
        sal_fname = target_filename(c_fname, args.prefix, ".sal")
        sal_bz_fname = target_filename(c_fname, args.prefix, ".sal.bz2")
        if not os.path.exists(as_fname) and not os.path.exists(sal_bz_fname):
            # Extract file
            tar.extract(tar_info)
            # Compile file
            print("APISAN " + c_fname)
            os.system(apisan + " compile " + c_fname + " 2> /dev/null > /dev/null")
            if not os.path.exists(as_fname):
                print("ERROR " + c_fname)
                continue
            # Remove filename
            os.unlink(c_fname)
            o_file = os.path.splitext(os.path.basename(c_fname))[0] + ".o"
            if os.path.exists(o_file):
                os.unlink(o_file)
            # Cleanup
            tar.members = []

        if not os.path.exists(sal_bz_fname):
            print("SAN2SAL " + as_fname)
            os.system("python3 " + as2sal + " -i " + as_fname + " -o " + sal_fname)
            os.unlink(as_fname)
            print("BZIP2 " + sal_fname)
            os.system("bzip2 " + sal_fname)
        else:
            print("OK " + sal_bz_fname)

if __name__ == '__main__':
    main()
