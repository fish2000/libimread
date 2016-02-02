
from unittest2 import TestCase
import sys
from os import listdir
from os.path import join, abspath, expanduser, dirname

class BaseCase(TestCase):
    
    def setUp(self):
        self._image_paths = map(
            lambda nm: expanduser(join(dirname(dirname(dirname(dirname(__file__)))), 'tests', 'data', nm)), filter(
                lambda nm: nm.lower().endswith('jpg'),
                    listdir(abspath(expanduser(join(dirname(dirname(dirname(dirname(__file__)))), 'tests', 'data'))))))
        self.image_paths = set(self._image_paths)


# class BaseCase(TestCase):
#
#     def setUp(self):
#         self._image_paths = map(
#             lambda nm: expanduser(join(dirname(__file__), 'testdata', 'img', nm)), filter(
#                 lambda nm: nm.lower().endswith('jpg'),
#                     listdir(abspath(expanduser(join(dirname(__file__), 'testdata', 'img'))))))[:17]
#         self.image_paths = set(self._image_paths)
#         self.imgc = map(
#             lambda image_path: imgc.PyCImage(image_path,
#                 dtype=imgc.uint8), self.image_paths)

def main(config=None):
    import nose
    return nose.main()

if __name__ == '__main__':
    sys.exit(main())