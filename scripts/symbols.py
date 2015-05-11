#!/usr/bin/env python

from __future__ import print_function

import re
from itertools import ifilter, imap, product
from os import walk
from os.path import join, dirname


SUFFIXES = (
    ".c",
    ".cc", ".cpp", ".cxx", "c++",
    ".m", ".mm",
    ".h",
    ".hh", ".hpp", ".hxx", ".h++",
    ".inl"
)

suffix_res = (re.compile(r"%s$" % suffix) for suffix in suffixes)

def collect_files(root_dir, *suffixes):
    """ Collect files recursively that match one or more file suffixes.
    """
    collected = set()
    print(list(suffixes))
    for pth, dirs, files in walk(root_dir):
        print("> Scanning directory %s (%i files)..." % (pth, len(files)))
        
        found = set(imap(lambda file, regex: join(pth, file),
            ifilter(lambda file, regex: regex.search(file, re.IGNORECASE),
                product(files, suffix_regexes))))
        print("> Collected %i of %i files" % (len(found), len(files)))
        collected |= found
    return collected

sanitizer = re.compile(r'^\W+')

def sanitize_suffixes(*suffixes):
    return (sanitizer.sub('', suffix.lower()) for suffix in suffixes)

if __name__ == '__main__':
    print("")
    print("> SCANNING...")
    print("")
    
    collected = collect_files(dirname(dirname(__file__)), *list(sanitize_suffixes(*SUFFIXES)))
    print("> FOUND %i grand total" % len(collected))
    
    for source_file in collected:
        print(">>> %s" % source_file)
    