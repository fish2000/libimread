
from __future__ import print_function
from unittest2 import TestCase

import im

class StructCodeTests(TestCase):
    
    def test_simple_structcodes(self):
        im.structcode_parse('B')
        im.structcode_parse('b')
        im.structcode_parse('Q')
        im.structcode_parse('O')
        im.structcode_parse('x')
        im.structcode_parse('d')
        im.structcode_parse('f')
    
    def test_less_simple_structcodes(self):
        im.structcode_parse('>BBBB')
        im.structcode_parse('=bb')
        im.structcode_parse('@QBQB')
        im.structcode_parse('OxOxO')
        im.structcode_parse('>??i')
        im.structcode_parse('efZfZd')
    
    def test_structcode_labels(self):
        im.structcode_parse('B:r: B:g: B:b:')             # RGB 888
        im.structcode_parse('d:X: d:Y: d:Z:')             # XYZ triple-dub
        im.structcode_parse('4f')                         # CMYK (unlabled)
        im.structcode_parse('xfxfxfxf')                   # CMYK (padded)
        im.structcode_parse('xf:C: xf:M: xf:Y: xf:K:')    # CMYK (everything)
