
from __future__ import print_function

from unittest2 import TestCase
import sys
from os import listdir
from os.path import join, abspath, expanduser, dirname, basename

class BaseCase(TestCase):
    
    def setUp(self):
        # imports
        import im
        from collections import namedtuple
        
        # get and store all paths
        self._image_paths = map(
            lambda nm: expanduser(join(dirname(dirname(dirname(dirname(__file__)))), 'tests', 'data', nm)),
                listdir(abspath(expanduser(join(dirname(dirname(dirname(dirname(__file__)))), 'tests', 'data')))))
        
        # filter paths, per file extension
        self.image_paths = set([pth for pth in self._image_paths if pth.lower().endswith('jpg')])
        self.jpgs = set([pth for pth in self._image_paths if pth.lower().endswith('jpg')])
        self.pngs = set([pth for pth in self._image_paths if pth.lower().endswith('png')])
        self.tifs = set([pth for pth in self._image_paths if basename(pth).lower().startswith('re')])
        self.pvrs = set([pth for pth in self._image_paths if pth.lower().endswith('pvr')])
        
        # from pprint import pformat
        # print(pformat(self.tifs, indent=4))
        
        # store local references to im.Image and im.Array
        ImageTypeTuple = namedtuple('ImageTypes', ['Image', 'Array'])
        self.imagetypes = ImageTypeTuple(im.Image, im.Array)
        

def main(discover=False):
    import nose2
    if discover:
        return nose2.discover()
    else:
        return nose2.main()

if __name__ == '__main__':
    sys.exit(main())