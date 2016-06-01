
from __future__ import print_function
from basecase import BaseCase

import numpy

class ReadTests(BaseCase):
    
    def _check_path(self, pth):
        from os.path import basename
        return not (basename(pth).lower().startswith("rgb") or \
                    basename(pth).lower().startswith("apple"))
    
    def check_path(self, pth):
        from os.path import basename
        return not basename(pth).lower().startswith("apple")
    
    def test_load_jpg(self):
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
                self.assertEqual(image.strides, image.buffer.strides)
    
    def test_load_png(self):
        for ImageType in self.imagetypes:
            for image_path in self.pngs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
                self.assertEqual(image.strides, image.buffer.strides)
    
    def test_load_tif(self):
        for ImageType in self.imagetypes:
            for image_path in self.tifs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
                self.assertEqual(image.strides, image.buffer.strides)
    
    def test_load_pvr(self):
        for ImageType in self.imagetypes:
            for image_path in self.pvrs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                self.assertEqual(image.shape, image.buffer.shape)
                self.assertEqual(image.strides, image.buffer.strides)
    
    def test_load_with_options(self):
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path, options=dict(yo="dogg"))
                self.assertIsNotNone(image)
                self.assertEqual(image.read_opts['yo'], "dogg")
                image.read_opts = dict(iheard="you like options dicts")
                self.assertFalse('yo' in image.read_opts)
                self.assertEqual(image.read_opts['iheard'], "you like options dicts")
    
    def test_load_jpg_check_dimensions(self):
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path)
                array = numpy.array(image)
                self.assertEqual(image.shape,   array.shape)
                self.assertEqual(image.strides, array.strides)
                self.assertEqual(len(image),    array.size)
                self.assertEqual(image.buffer.shape,   array.shape)
                self.assertEqual(image.buffer.strides, array.strides)
                self.assertEqual(len(image.buffer),    array.size)
    
    def test_load_png_check_dimensions(self):
        for ImageType in self.imagetypes:
            for image_path in self.pngs:
                image = ImageType(image_path)
                array = numpy.array(image)
                self.assertEqual(image.shape,   array.shape)
                self.assertEqual(image.strides, array.strides)
                self.assertEqual(len(image),    array.size)
                self.assertEqual(image.buffer.shape,   array.shape)
                self.assertEqual(image.buffer.strides, array.strides)
                self.assertEqual(len(image.buffer),    array.size)
    
    def test_load_tif_check_dimensions(self):
        for ImageType in self.imagetypes:
            for image_path in self.tifs:
                image = ImageType(image_path)
                array = numpy.array(image)
                self.assertEqual(image.shape,   array.shape)
                self.assertEqual(image.strides, array.strides)
                self.assertEqual(len(image),    array.size)
                self.assertEqual(image.buffer.shape,   array.shape)
                self.assertEqual(image.buffer.strides, array.strides)
                self.assertEqual(len(image.buffer),    array.size)
    
    def test_load_jpg_as_blob(self):
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                with open(image_path, 'rb') as image_fh:
                    image_blob = image_fh.read()
                    image = ImageType(image_blob, is_blob=True)
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_png_as_blob(self):
        for ImageType in self.imagetypes:
            for image_path in self.pngs:
                with open(image_path, 'rb') as image_fh:
                    image_blob = image_fh.read()
                    image = ImageType(image_blob, is_blob=True)
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_tif_as_blob(self):
        for ImageType in self.imagetypes:
            for image_path in self.tifs:
                with open(image_path, 'rb') as image_fh:
                    image_blob = image_fh.read()
                    image = ImageType(image_blob, is_blob=True)
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_pvr_as_blob(self):
        ''' Need to ensure Apple PVR data is recognized '''
        for ImageType in self.imagetypes:
            for image_path in self.pvrs:
                if self.check_path(image_path):
                    with open(image_path, 'rb') as image_fh:
                        image_blob = image_fh.read()
                        image = ImageType(image_blob, is_blob=True)
                        self.assertIsNotNone(image)
                        self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_jpg_as_blob_from_filehandle(self):
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                with open(image_path, 'rb') as image_fh:
                    image = ImageType(file=image_fh, is_blob=True)
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_png_as_blob_from_filehandle(self):
        for ImageType in self.imagetypes:
            for image_path in self.pngs:
                with open(image_path, 'rb') as image_fh:
                    image = ImageType(file=image_fh, is_blob=True)
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_tif_as_blob_from_filehandle(self):
        ''' This does not work for some reason -- likely due to the
            wackadoo source/sink manipulation in IO/tiff.cpp
        '''
        for ImageType in self.imagetypes:
            for image_path in self.tifs:
                with open(image_path, 'rb') as image_fh:
                    image = ImageType(file=image_fh, is_blob=True)
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape, image.buffer.shape)
    
    def test_load_pvr_as_blob_from_filehandle(self):
        for ImageType in self.imagetypes:
            for image_path in self.pvrs:
                with open(image_path, 'rb') as image_fh:
                    image = ImageType(file=image_fh, is_blob=True)
                    self.assertIsNotNone(image)
                    self.assertEqual(image.shape, image.buffer.shape)
