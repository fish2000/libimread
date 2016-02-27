#!/usr/bin/env python
from __future__ import print_function
import sys
from im.tests.basecase import main

sys.argv.append('--verbosity=2')
print(timeit.timeit("main()", setup="from im.tests.basecase import main"))