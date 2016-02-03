#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
#       sigbytes.py
#
#       Read and print the first 8 or so bytes of a file,
#       For, like, tests and stuff
#       (c) 2015 Alexander Bohn, All Rights Reserved
#
"""
Usage:
  sigbytes.py INFILE... [-o OUTFILE]
                        [-s SIZE]
                        [-V | --verbose]
  
  sigbytes.py -h | --help | -v | --version
    
Options:
  -o OUTFILE --output=OUTFILE       specify output file [default: stdout]
  -s SIZE    --size=SIZE            specify size of bytes to read [default: 8]
  -V --verbose                      print verbose output
  -h --help                         show this text
  -v --version                      print version
"""

from __future__ import print_function
from os.path import exists, isdir, dirname, expanduser
from docopt import docopt
import sys

def cli(argv=None):
    if not argv:
        argv = sys.argv
    
    arguments = docopt(__doc__, argv=argv[1:],
                                help=True,
                                version='0.1.0')
    
    # print(argv)
    # print(arguments)
    # sys.exit()
    
    ipths = (expanduser(pth) for pth in arguments.get('INFILE'))
    opth = expanduser(arguments.get('--output'))
    siz = int(arguments.get('-s', 8))
    verbose = bool(arguments.get('--verbose'))
    
    if not len(arguments.get('INFILE')) > 0:
        raise AttributeError("No input files")
    
    for ipth in ipths:
        fp = None
        if opth != 'stdout':
            if exists(opth) or isdir(dirname(opth)):
                raise AttributeError("Bad output file")
            else:
                fp = open(opth, 'wb')
        with open(ipth, 'rb') as fh:
            header_bytes = fh.read(siz)
            if opth == 'stdout':
                if verbose:
                    print(">>> Header bytes (%s) for %s" % (siz, ipth))
                    print(">>> %s" % repr(header_bytes))
                else:
                    print(repr(header_bytes)[1:-1])
            else:
                fp.write(header_bytes)
                fp.write("\n")
        if fp is not None:
            fp.flush()
            fp.close()

if __name__ == '__main__':
    cli(sys.argv)