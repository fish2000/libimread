
from __future__ import print_function
from basecase import BaseCase

class PlanarOperationTests(BaseCase):
    
    def test_jpg_split(self):
        ''' Load some JPG files, split into plane image tuple,
            compare plane features (size etc) to image features
        '''
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                planes = image.split()
                self.assertEqual(int(image.planes), len(planes))
                for plane in planes:
                    self.assertEqual(plane.width,  image.width)
                    self.assertEqual(plane.height, image.height)
                    self.assertEqual(plane.planes, 1)
    
    def test_jpg_split_merge(self):
        ''' Load some JPG files, split into plane image tuple,
            compare plane features (size etc) to image features,
            re-merge into composite image
        '''
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                planes = image.split()
                self.assertEqual(int(image.planes), len(planes))
                image2 = ImageType.merge(planes)
                self.assertEqual(image.width,  image2.width)
                self.assertEqual(image.height, image2.height)
                self.assertEqual(image.planes, image2.planes)
                self.assertEqual(image.buffer.tostring(),
                                 image2.buffer.tostring())
    
    def test_jpg_alpha(self):
        ''' Load some JPG files, check for an alpha channel --
            if alpha is present: check the mode and the number of planes;
            if not: create a new image with an alpha channel
                    from the image we loaded, check the mode
                    and the number of planes, and test the
                    new image width/height against the original.
        '''
        alpha_modes = ('LA', 'RGBA')
        alpha_planes = (2, 4)
        for ImageType in self.imagetypes:
            for image_path in self.jpgs:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                if not image.has_alpha:
                    # no alpha channel - try to add one:
                    alpha_image = image.add_alpha()
                    self.assertTrue(alpha_image.mode in alpha_modes)
                    self.assertTrue(alpha_image.planes in alpha_planes)
                    self.assertEqual(alpha_image.width, image.width)
                    self.assertEqual(alpha_image.height, image.height)
                else:
                    # alpha channel already present:
                    self.assertTrue(image.mode in alpha_modes)
                    self.assertTrue(image.planes in alpha_planes)
    