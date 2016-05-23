
from __future__ import print_function
from basecase import BaseCase

import im
import numpy

class ArrayReadTests(BaseCase):
    
    def test_load_array_jpg(self):
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, image.buffer.shape)
            self.assertEqual(image.strides, image.buffer.strides)
    
    def test_load_array_png(self):
        for image_path in self.pngs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, image.buffer.shape)
            self.assertEqual(image.strides, image.buffer.strides)
    
    def test_load_array_tif(self):
        for image_path in self.tifs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, image.buffer.shape)
            self.assertEqual(image.strides, image.buffer.strides)
    
    def test_load_array_with_options(self):
        for image_path in self.jpgs:
            image = im.Array(image_path, options=dict(yo="dogg"))
            self.assertIsNotNone(image)
            self.assertEqual(image.read_opts['yo'], "dogg")
            image.read_opts = dict(iheard="you like options dicts")
            self.assertFalse('yo' in image.read_opts)
            self.assertEqual(image.read_opts['iheard'], "you like options dicts")
    
    def test_load_array_jpg_check_dimensions(self):
        for image_path in self.jpgs:
            image = im.Array(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
            self.assertEqual(image.buffer.shape,   array.shape)
            self.assertEqual(image.buffer.strides, array.strides)
            self.assertEqual(len(image.buffer),    array.size)
    
    def test_load_array_png_check_dimensions(self):
        for image_path in self.pngs:
            image = im.Array(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
            self.assertEqual(image.buffer.shape,   array.shape)
            self.assertEqual(image.buffer.strides, array.strides)
            self.assertEqual(len(image.buffer),    array.size)
    
    def test_load_array_tif_check_dimensions(self):
        for image_path in self.tifs:
            image = im.Array(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
            self.assertEqual(image.buffer.shape,   array.shape)
            self.assertEqual(image.buffer.strides, array.strides)
            self.assertEqual(len(image.buffer),    array.size)
    
    def test_load_array_jpg_as_blob(self):
        for image_path in self.jpgs:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.Array(image_blob, is_blob=True)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_array_png_as_blob(self):
        for image_path in self.pngs:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.Array(image_blob, is_blob=True)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_array_tif_as_blob(self):
        for image_path in self.tifs:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.Array(image_blob, is_blob=True)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_array_jpg_as_blob_from_filehandle(self):
        for image_path in self.jpgs:
            with open(image_path, 'rb') as image_fh:
                image = im.Array(file=image_fh, is_blob=True)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_array_png_as_blob_from_filehandle(self):
        for image_path in self.pngs:
            with open(image_path, 'rb') as image_fh:
                image = im.Array(file=image_fh, is_blob=True)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
    
    """
    def test_load_array_tif_as_blob_from_filehandle(self):
        for image_path in self.tifs:
            with open(image_path, 'rb') as image_fh:
                image = im.Array(file=image_fh, is_blob=True)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
    """
