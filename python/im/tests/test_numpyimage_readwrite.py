
from __future__ import print_function
from basecase import BaseCase

import im
# import numpy

class NumpyImageReadWriteTests(BaseCase):
    
    def test_numpyimage_check(self):
        # ''' Re-using `test_load_image_jpg_write_blob_jpg` here '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            self.assertTrue(im.numpyimage_check(image))
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
            self.assertFalse(im.numpyimage_check(data))
    
    def test_load_image_jpg_write_blob_jpg(self):
        ''' Load JPG files, write JPG blobs '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
    
    def test_load_image_png_write_blob_jpg(self):
        ''' Load PNG files, write JPG blobs '''
        for image_path in self.pngs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
    
    def test_load_image_jpg_write_blob_png(self):
        ''' Load JPG files, write PNG blobs '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
    
    def test_load_image_png_write_blob_png(self):
        ''' Load PNG files, write PNG blobs '''
        for image_path in self.pngs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
    
    def test_load_image_jpg_write_blob_tif(self):
        ''' Load JPG files, write TIF blobs '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "tif" })
            self.assertIsNotNone(data)
    
    def test_load_image_jpg_write_blob_png_readback(self):
        ''' Load JPG files, write PNG blobs with readback '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
            image2 = im.NumpyImage(data, is_blob=True)
            self.assertIsNotNone(image2)
            self.assertEqual(image.shape, image2.shape)
    
    def test_load_image_png_write_blob_png_readback(self):
        ''' Load PNG files, write PNG blobs with readback '''
        for image_path in self.pngs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
            image2 = im.NumpyImage(data, is_blob=True)
            self.assertIsNotNone(image2)
            self.assertEqual(image.shape[:2], image2.shape[:2])
    
    def test_load_image_jpg_write_blob_jpg_readback(self):
        ''' Load JPG files, write PNG blobs with readback '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
            image2 = im.NumpyImage(data, is_blob=True)
            self.assertIsNotNone(image2)
            self.assertEqual(image.shape, image2.shape)
    
    def test_load_image_png_write_blob_jpg_readback(self):
        ''' Load PNG files, write PNG blobs with readback '''
        for image_path in self.pngs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
            image2 = im.NumpyImage(data, is_blob=True)
            self.assertIsNotNone(image2)
            self.assertEqual(image.shape[:2], image2.shape[:2])
    
    def test_load_image_jpg_write_blob_tif_readback(self):
        ''' Load JPG files, write TIF blobs with readback '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "tif" })
            self.assertIsNotNone(data)
            image2 = im.NumpyImage(data, is_blob=True)
            self.assertIsNotNone(image2)
            # self.assertEqual(image.shape[:2], image2.shape[:2])
    
    def test_load_image_jpg_write_blob_tif_options_readback(self):
        ''' Load JPG files, write TIF blobs with readback + metadata options '''
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "tif", 'tiff:metadata'            : True,
                                                                         'metadata'                 : "YO DOGG",
                                                                         'tiff:software-signature'  : "I HEARD YOU LIKE METADATA" })
            self.assertIsNotNone(data)
            image2 = im.NumpyImage(data, is_blob=True)
            self.assertIsNotNone(image2)
            # self.assertEqual(image.shape[:2], image2.shape[:2])
    
    '''
    def test_load_image_jpg_check_dimensions(self):
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
    '''
    
    '''
    def test_load_image_png_check_dimensions(self):
        for image_path in self.pngs:
            image = im.NumpyImage(image_path)
            array = numpy.array(image)
            self.assertEqual(image.shape,   array.shape)
            self.assertEqual(image.strides, array.strides)
            self.assertEqual(len(image),    array.size)
    '''
    
    '''
    def test_load_image_jpg_as_blob(self):
        for image_path in self.jpgs:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.NumpyImage(image_blob, is_blob=True)
                self.assertIsNotNone(image)
    '''
    
    '''
    def test_load_image_png_as_blob(self):
        for image_path in self.pngs:
            with open(image_path, 'rb') as image_fh:
                image_blob = image_fh.read()
                image = im.NumpyImage(image_blob, is_blob=True)
                self.assertIsNotNone(image)
    '''