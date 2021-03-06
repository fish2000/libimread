# -*- coding: utf-8 -*-
# Copyright (C) 2012-2016, Alexander Böhn <fish2000@gmail.com>
# License: MIT (see COPYING.MIT file)

"""
_imread.py: libimread shim

This module is a shim, providing the functions that @luispedro imread
furnishes in the _imread.so extension module -- the function signatures
are legible in _imread.cpp within that codebase.

This shim module uses numpy and libimread's Python extension module to
perform the necessary functions. 

At the time of writing, the *_multi functions aren't available from Python.
"""

from __future__ import print_function

import im, numpy

all_suffixes = set()
for format_name, format_opts in im.format_info.iteritems():
    all_suffixes |= set(format_opts.get('suffixes'))

def imread(filename, formatstr, opts):
    return (numpy.array(im.Array(filename, options=opts)).transpose(1, 0, 2), '')

def imread_from_blob(blob, formatstr, opts):
    return (numpy.array(im.Array(blob, is_blob=True, options=opts)).transpose(1, 0, 2), '')

def imread_multi(filename, formatstr, opts):
    raise NotImplementedError(
        "_imread.imread_multi is as-of-yet unsupported")

def imsave(filename, formatstr, array, opts):
    imbuffer = im.Buffer.frompybuffer(array.transpose(1, 0, 2))
    image = im.Array.frombuffer(imbuffer)
    image.write(filename, as_blob=False, options=opts)

def imsave_multi(filename, formatstr, arrays, opts):
    raise NotImplementedError(
        "_imread.imsave_multi is as-of-yet unsupported")

def detect_format(filename_or_blob, is_blob=False):
    return im.detect(filename_or_blob, is_blob=bool(is_blob))

def supports_format(formatstr):
    return formatstr.lower() in all_suffixes