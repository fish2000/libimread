#!/usr/bin/env python
import sys
from im.tests.basecase import main
sys.argv.append('--verbosity=2')
sys.exit(main())