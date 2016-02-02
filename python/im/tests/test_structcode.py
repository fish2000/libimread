
from __future__ import print_function
# from unittest2 import TestCase
from basecase import BaseCase

import im

class StructCodeTests(BaseCase):
    
    @staticmethod
    def singletup(dtypechar):
        return (('f0', dtypechar),)
    
    @staticmethod
    def multitup(*chars):
        out = []
        for idx, char in enumerate(chars):
            out.append(('f%i' % idx, char))
        return tuple(out)
    
    def test_simple_structcodes(self):
        self.assertEqual(im.structcode_parse('B'),
                         self.singletup('B'))
        self.assertEqual(im.structcode_parse('b'),
                         self.singletup('b'))
        self.assertEqual(im.structcode_parse('Q'),
                         self.singletup('u8'))
        self.assertEqual(im.structcode_parse('O'),
                         self.singletup('O'))
        self.assertEqual(im.structcode_parse('x'),
                         self.singletup('V'))
        self.assertEqual(im.structcode_parse('d'),
                         self.singletup('d'))
        self.assertEqual(im.structcode_parse('f'),
                         self.singletup('f'))
    
    def test_less_simple_structcodes(self):
        self.assertEqual(im.structcode_parse('>BBBB'),
                         self.multitup('>B', '>B', '>B', '>B'))
        self.assertEqual(im.structcode_parse('=bb'),
                         self.multitup('=b', '=b'))
        self.assertEqual(im.structcode_parse('@QBQB'),
                         self.multitup('=u8', '=B', '=u8', '=B'))
        self.assertEqual(im.structcode_parse('OxOxO'),
                         self.multitup('O', 'V', 'O', 'V', 'O'))
        self.assertEqual(im.structcode_parse('=??i'),
                         self.multitup('=?', '=?', '=i4'))
        self.assertEqual(im.structcode_parse('efZfZd'),
                         self.multitup('f2', 'f', 'F', 'D'))
    
    def test_structcode_labels(self):
        
        # RGB 888
        self.assertEqual(
            im.structcode_parse('B:r: B:g: B:b:'),
            (('r', 'B'), ('g', 'B'), ('b', 'B'))
        )
        
        # XYZ triple-dub
        self.assertEqual(
            im.structcode_parse('d:X: d:Y: d:Z:'),
            (('X', 'd'), ('Y', 'd'), ('Z', 'd'))
        )
        
        # CMYK (unlabled float quadruple)
        self.assertEqual(
            im.structcode_parse('4f'),
            (('f0', '4f'),)
        )
        
        # CMYK (float values with padding)
        self.assertEqual(
            im.structcode_parse('xfxfxfxf'),
            (('f0', 'V'), ('f1', 'f'), ('f2', 'V'), ('f3', 'f'),
             ('f4', 'V'), ('f5', 'f'), ('f6', 'V'), ('f7', 'f'))
        )
        
        # CMYK (everything: labeled padded float values)
        self.assertEqual(
            im.structcode_parse('xf:C: xf:M: xf:Y: xf:K:'),
            (('f0', 'V'), ('C', 'f'),
             ('f1', 'V'), ('M', 'f'),
             ('f2', 'V'), ('Y', 'f'),
             ('f3', 'V'), ('K', 'f'))
        )
        
        
