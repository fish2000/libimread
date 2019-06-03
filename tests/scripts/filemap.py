#!/usr/bin/env python
#
#       filemap.py
#
#       Sort a list of files into a dict of separate lists by suffix
#       For, like, tests and stuff
#       (c) 2015-2019 Alexander Bohn, All Rights Reserved
#

from __future__ import print_function, unicode_literals

def filemapper(files):
    """ Sort a list of files into a dict of separate lists by suffix """
    return dict(
        jpg    = [f for f in files if f.endswith('jpg')],
        jpeg   = [f for f in files if f.endswith('jpeg')],
        png    = [f for f in files if f.endswith('png')],
        tif    = [f for f in files if f.endswith('tif')],
        tiff   = [f for f in files if f.endswith('tiff')],
        hdf5   = [f for f in files if f.endswith('hdf5')],
        pvr    = [f for f in files if f.endswith('pvr')])