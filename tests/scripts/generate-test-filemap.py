#!/usr/bin/env python
#
#       geerate-test-filemap.py
#
#       Scan test data folder and dump discovered image files,
#           grouped by file type, to a serializable format
#           e.g. property list, msgpack, json etc
#       For, like, tests and stuff
#       (c) 2015-2019 Alexander Bohn, All Rights Reserved
#

from __future__ import print_function, unicode_literals
import sys, os
from filemap import filemapper

datadir = "../data"

def main():
    os.chdir(os.path.dirname(__file__))
    os.chdir(datadir)
    basedir = os.getcwd()
    files = os.listdir(basedir)
    
    filemap = filemapper(files)
    filemap['basedir'] = basedir
    
    if 'json' in sys.argv:
        # print json (prettily)
        import json
        print(json.dumps(filemap, indent=4))
    elif 'pack' in sys.argv or 'msgpack' in sys.argv:
        # print msgpack (compact)
        import msgpack
        print(msgpack.dumps(filemap))
    else:
        # print xml property list 
        import plistlib
        if hasattr(plistlib, 'dumps'):
            print(plistlib.dumps(filemap))
        if hasattr(plistlib, 'writePlistToString'):
            print(plistlib.writePlistToString(filemap))
        raise NameError("Can't find plistlib write-to-string method")

if __name__ == '__main__':
    main()
