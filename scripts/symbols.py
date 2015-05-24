#!/usr/bin/env python

from __future__ import print_function

import re
import functools
from itertools import ifilter, imap
from os import walk
from os.path import join, dirname
# from pprint import pprint

SUFFIXES = set((
    r"c",
    r"cc", r"cpp", r"cxx",
    r"m", r"mm",
    r"h",
    r"hh", r"hpp", r"hxx",
    r"inl"
))

KEYWORDS = set((
    "alignas", "alignof", "and", "and_eq", "asm", "auto",
    "bitand", "bitor", "bool", "break",
    "case", "catch", "char", "char16_t", "char32_t", "class",
    "compl", "const", "constexpr", "const_cast", "continue",
    "decltype", "default", "delete", "do", "double", "dynamic_cast",
    "else", "enum", "explicit", "export", "extern",
    "false", "float", "for", "friend",
    "goto",
    "if", "inline", "int",
    "long",
    "mutable",
    "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
    "operator", "or", "or_eq",
    "private", "protected", "public",
    "register", "reinterpret_cast", "return",
    "short", "signed", "sizeof", "static", "static_assert",
    "static_cast", "struct", "switch",
    "template", "this", "thread_local", "throw", "true",
    "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using",
    "virtual", "void", "volatile",
    "wchar_t", "while",
    "xor", "xor_eq"))

SYMBOL_PREFIX = r'_'

sanitizer = re.compile(r'^\W+')
# symbol_re = re.compile(r"(?:\w+)%s([a-zA-Z][0-9a-zA-Z_]+)(?!_t)" % SYMBOL_PREFIX)
symbol_re = re.compile(r"(?:\s+)(?:%s)([a-z][0-9a-z_]+)(?!_t)" % SYMBOL_PREFIX)
suffix_re = re.compile(r"(%s)$" % ("|".join(map(
    lambda suffix: r"\.%s" % suffix, SUFFIXES))))

quotes          = re.compile(r"'(.*)'",     re.MULTILINE)
dbl_quotes      = re.compile(r'"(.*)"',     re.MULTILINE)
cpp_comments    = re.compile(r'//(.*)\n',   re.MULTILINE)
c_comments      = re.compile(r'/\*(.*)\*/', re.MULTILINE)
blockout        = lambda match: "#" * len(match.group(0))

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

def sanitize_suffixes(*suffixes):
    return (sanitizer.sub('', suffix.lower()) for suffix in suffixes)

collect = functools.partial(collect_files, suffixes=list(sanitize_suffixes(*SUFFIXES)))

def parse_source_file(fn):
    with open(fn, "rb") as fh:
        source_file = fh.read()
    source_edit = (source_file,)
    for blockout_re in (quotes, dbl_quotes, cpp_comments, c_comments):
        source_edit = blockout_re.subn(blockout, source_edit[0])
    try:
        sym = set(symbol_re.findall(source_edit[0]))
    except:
        return set()
    return sym - KEYWORDS

if __name__ == '__main__':
    print("")
    print("> SCANNING...")
    print("")
    
    symbols = set()
    collected = collect(dirname(dirname(__file__)))
    
    for source_file in sorted(collected):
        print("Parsing file: %s" % source_file)
        symbols |= parse_source_file(source_file)
    
    print("> FOUND %i grand total:" % len(symbols))
    # print(symbols)
    print(", ".join(sorted(symbols)))
    