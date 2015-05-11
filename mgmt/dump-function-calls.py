#!/usr/bin/env python
# dump_function_calls.py:
#   Originally from: https://github.com/smspillaz/scripts/blob/master/list_non_inlined_symbols/dump_function_calls.py
#
# Copyright (c) 2014 Sam Spilsbury <smspillaz@gmail.com>
# Licenced under the MIT Licence.
#
# Looks at the DWARF data for a library and dumps to stdout where
# functions are called
#
# Usage: dump_function_calls.py object [regex]

import re
import sys
import subprocess

def get_function_calls (objdump_output, regex):
    function_calls = []

    for line in objdump_output.split ("\n"):
        if "callq" in line and "<" in line and ">" in line:
            if regex is None or (regex is not None and regex.match (line) != None):
                mangled = line.split ("<")[1]
                if "@" in mangled:
                    mangled = mangled.split("@")[0]
                elif "." in mangled:
                    mangled = mangled.split(".")[0]
                call = subprocess.check_output (["c++filt", mangled])[:-1]
                function_calls.append (call)

    return set (function_calls)

if (len (sys.argv) < 2):
    print "Usage: dump_function_calls.py object [regex]"

object = sys.argv[1];
regex = None

if (len (sys.argv) == 3):
    regex = re.compile (sys.argv[2])

objdump_output = subprocess.check_output (["gobjdump", "-S", object])
function_calls = get_function_calls (objdump_output, regex)

for call in function_calls:
    print call