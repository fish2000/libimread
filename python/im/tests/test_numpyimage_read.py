
from __future__ import print_function
from basecase import BaseCase

import im
import numpy

class HybridImageReadTests(BaseCase):
    
    def test_load_image_jpg(self):
        for image_path in self.jpgs:
            image = im.HybridImage(image_path)
            self.assertIsNotNone(image)
    
    def test_load_image_png(self):
        for image_path in self.pngs:
            image = im.HybridImage(image_path)
            self.assertIsNotNone(image)
    
    def test_load_image_with_options(self):
        for image_path in self.jpgs:
            image = im.HybridImage(image_path, options=dict(yo="dogg"))
            self.assertIsNotNone(image)
            self.assertEqual(image.read_opts['yo'], "dogg")
            image.read_opts = dict(iheard="you like options dicts")
            self.assertFalse('yo' in image.read_opts)
            self.assertEqual(image.read_opts['iheard'], "you like options dicts")
    
    def test_load_image_jpg_check_dimensions(self):
        for image_path in self.jpgs:
            image = im.HybridImage(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
    
    def test_load_image_png_check_dimensions(self):
        for image_path in self.pngs:
            image = im.HybridImage(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
    
    def test_load_image_jpg_as_blob(self):
        for image_path in self.jpgs:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.HybridImage(image_blob, is_blob=True)
                self.assertIsNotNone(image)
    
    def test_load_image_png_as_blob(self):
        for image_path in self.pngs:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.HybridImage(image_blob, is_blob=True)
                self.assertIsNotNone(image)
    