
from __future__ import print_function
from basecase import BaseCase

import im

class CompatibilityTests(BaseCase):
    
    def test_imread_imread(self):
        import numpy
        import imread as luispedro
        from im.compat import imread
        for image_path in self.jpgs:
            ar0 = imread.imread(image_path)
            ar1 = luispedro.imread(image_path)
            self.assertTrue(numpy.all(ar0 == ar1))
    
    def test_imread_imread_from_blob(self):
        import numpy
        import imread as luispedro
        from im.compat import imread
        for image_path in self.jpgs:
            with open(image_path, 'rb') as fh:
                blob = fh.read()
                ar0 = imread.imread_from_blob(blob)
                ar1 = luispedro.imread_from_blob(blob)
                self.assertTrue(numpy.all(ar0 == ar1))
    
    def test_imread_detect_format(self):
        import imread as luispedro
        from im.compat import imread
        for image_path in self.jpgs:
            format0 = imread.detect_format(image_path)
            format1 = luispedro.detect_format(image_path)
            self.assertEqual(format0, format1)
            with open(image_path, 'rb') as fh:
                blob = fh.read()
                blobformat0 = imread.detect_format(blob, is_blob=True)
                blobformat1 = luispedro.detect_format(blob, is_blob=True)
                self.assertEqual(blobformat0, blobformat1)
    
    def test_jpg_convert_to_PIL(self):
        ''' Load some JPG files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path)
                # self.assertIsNotNone(image)
                pil_image = im.to_PIL(image)
                # self.assertIsNotNone(pil_image)
                self.assertEqual(pil_image.width,  image.width)
                self.assertEqual(pil_image.height, image.height)
    
    def test_png_convert_to_PIL(self):
        ''' Load some PNG files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.pngs:
                image = ImageType(image_path)
                # self.assertIsNotNone(image)
                pil_image = im.to_PIL(image)
                # self.assertIsNotNone(pil_image)
                self.assertEqual(pil_image.width,  image.width)
                self.assertEqual(pil_image.height, image.height)
    
    def _test_tif_convert_to_PIL(self):
        ''' Load some TIF files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.tifs:
                image = ImageType(image_path)
                # self.assertIsNotNone(image)
                pil_image = im.to_PIL(image, mode="RGB")
                self.assertIsNotNone(pil_image)
                ''' BUGGGG: TIFs are still fucking transposed '''
                # self.assertEqual(pil_image.width,  image.width)
                # self.assertEqual(pil_image.height, image.height)
    
    def test_pvr_convert_to_PIL(self):
        ''' Load some PVR files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.pvrs:
                image = ImageType(image_path)
                # self.assertIsNotNone(image)
                pil_image = im.to_PIL(image)
                # self.assertIsNotNone(pil_image)
                self.assertEqual(pil_image.width,  image.width)
                self.assertEqual(pil_image.height, image.height)
    