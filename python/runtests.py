#!/usr/bin/env python
import sys
from im.tests.basecase import main
sys.argv.append('--verbose')
sys.exit(main(discover=True))