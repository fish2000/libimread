
from __future__ import print_function
from basecase import BaseCase

import im

class CompatibilityTests(BaseCase):
    
    def test_jpg_convert_to_PIL(self):
        ''' Load some JPG files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                pil_image = im.to_PIL(image)
                self.assertIsNotNone(pil_image)
                self.assertEqual(pil_image.width,  image.width)
                self.assertEqual(pil_image.height, image.height)
    
    def test_png_convert_to_PIL(self):
        ''' Load some PNG files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.pngs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                pil_image = im.to_PIL(image)
                self.assertIsNotNone(pil_image)
                self.assertEqual(pil_image.width,  image.width)
                self.assertEqual(pil_image.height, image.height)
    
    """
    def test_tif_convert_to_PIL(self):
        ''' Load some PNG files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.tifs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                pil_image = im.to_PIL(image, mode="RGB")
                self.assertIsNotNone(pil_image)
                ''' BUGGGG: TIFs are still fucking transposed '''
                # self.assertEqual(pil_image.width,  image.width)
                # self.assertEqual(pil_image.height, image.height)
    """
    
    def test_pvr_convert_to_PIL(self):
        ''' Load some PNG files,
            convert to PIL.Image instances,
            compare image features (size etc) '''
        for ImageType in self.imagetypes:
            for image_path in self.pvrs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                pil_image = im.to_PIL(image)
                self.assertIsNotNone(pil_image)
                self.assertEqual(pil_image.width,  image.width)
                self.assertEqual(pil_image.height, image.height)
    