#!/usr/bin/env python

from __future__ import print_function

import re
import functools
from itertools import ifilter, imap
from os import walk
from os.path import join, dirname
# from pprint import pprint

SUFFIXES = (
    r"c",
    r"cc", r"cpp", r"cxx",
    r"m", r"mm",
    r"h",
    r"hh", r"hpp", r"hxx",
    r"inl"
)

suffix_re = re.compile(r"(%s)$" % ("|".join(map(
    lambda suffix: r"\.%s" % suffix, SUFFIXES))))

def collect_files(root_dir, suffixes):
    """ Collect files recursively that match one or more file suffixes.
    """
    collected = set()
    print(list(suffixes))
    for pth, dirs, files in walk(root_dir):
        print("> Scanning directory %s (%i files)..." % (pth, len(files)))
        # pprint(list(product(files, suffix_res)))
        found = set(imap(lambda file: join(pth, file),
            ifilter(lambda file: suffix_re.search(file, re.IGNORECASE), files)))
        print("> Collected %i of %i files" % (len(found), len(files)))
        collected |= found
    return collected

sanitizer = re.compile(r'^\W+')

def sanitize_suffixes(*suffixes):
    return (sanitizer.sub('', suffix.lower()) for suffix in suffixes)

collect = functools.partial(collect_files, suffixes=list(sanitize_suffixes(*SUFFIXES)))

if __name__ == '__main__':
    print("")
    print("> SCANNING...")
    print("")
    
    # collected = collect_files(dirname(dirname(__file__)), suffixes=list(sanitize_suffixes(*SUFFIXES)))
    collected = collect(dirname(dirname(__file__)))
    print("> FOUND %i grand total" % len(collected))
    
    for source_file in sorted(collected):
        print(">>> %s" % source_file)
    