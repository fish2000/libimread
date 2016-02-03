
from __future__ import print_function
from basecase import BaseCase

import im
import numpy

class NumpyImageTests(BaseCase):
    
    def test_load_image(self):
        for image_path in self.image_paths:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
    
    def test_load_image_with_options(self):
        for image_path in self.image_paths:
            image = im.NumpyImage(image_path, options=dict(yo="dogg"))
            self.assertIsNotNone(image)
            self.assertEqual(image.read_opts['yo'], "dogg")
            image.read_opts = dict(iheard="you like options dicts")
            self.assertFalse('yo' in image.read_opts)
            self.assertEqual(image.read_opts['iheard'], "you like options dicts")
    
    def test_load_image_check_dimensions(self):
        for image_path in self.image_paths:
            image = im.NumpyImage(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
    
    def test_load_image_blob(self):
        for image_path in self.image_paths:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.NumpyImage(image_blob, is_blob=True)
                self.assertIsNotNone(image)
    