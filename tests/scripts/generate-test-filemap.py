#!/usr/bin/env python
#
#       geerate-test-filemap.py
#
#       Scan test data folder and dump discovered image files,
#           grouped by file type, to a serializable format
#           e.g. property list, msgpack, json etc
#       For, like, tests and stuff
#       (c) 2015 Alexander Bohn, All Rights Reserved
#

from __future__ import print_function
from os import chdir, listdir, getcwd
from os.path import dirname
from filemap import filemapper

datadir = "../data"

if __name__ == '__main__':
    import sys
    
    chdir(dirname(__file__))
    chdir(datadir)
    basedir = getcwd()
    files = listdir(basedir)
    
    filetpls = dict()
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
        print(plistlib.writePlistToString(filemap))




