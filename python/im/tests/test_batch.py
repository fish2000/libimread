
from __future__ import print_function
from basecase import BaseCase

import im
# import numpy
from im.compat import imread

class BatchTests(BaseCase):
    
    def test_batch_iter_jpgs(self):
        for ImageType in self.imagetypes:
            batch0 = im.Batch()
            for image_path in self.jpgs:
                batch0.append(ImageType(image_path))
            batch1 = im.Batch()
            for image in batch0:
                batch1.append(image)
            self.assertEqual(len(batch0), len(batch1))
            # TODO: assertRaises for batch.width / batch.height
    
    def test_batch_load_jpgs(self):
        for image_path in self.jpgs:
            image0 = self.imagetypes.Image(image_path)
            image1 = self.imagetypes.Image(image_path)
            batch0 = im.Batch(image0, image1)
            self.assertEqual(len(batch0), 2)
            self.assertEqual(batch0.width,  image0.width)
            self.assertEqual(batch0.height, image0.height)
            self.assertEqual(batch0.width,  image1.width)
            self.assertEqual(batch0.height, image1.height)
            batch1 = im.Batch()
            batch1.append(self.imagetypes.Array(image_path))
            batch1.append(self.imagetypes.Array(image_path))
            self.assertEqual(len(batch0), len(batch1))
            batch1.append(imread.imread(image_path))
            self.assertFalse(batch0 == batch1)
    
    def test_batch_load_pngs(self):
        for image_path in self.pngs:
            image = self.imagetypes.Image(image_path)
            array = self.imagetypes.Array(image_path)
            batch0 = self.Batch(image, array)
            self.assertEqual(len(batch0), 2)
            self.assertEqual(batch0.width,  image.width)
            self.assertEqual(batch0.height, image.height)
            self.assertEqual(batch0.width,  array.width)
            self.assertEqual(batch0.height, array.height)
            batch1 = self.Batch()
            batch1.append(self.imagetypes.Image(image_path))
            batch1.append(self.imagetypes.Array(image_path))
            self.assertEqual(len(batch0), len(batch1))
            batch1.append(imread.imread(image_path))
            self.assertFalse(batch0 == batch1)
    
    def test_batch_load_tifs(self):
        for image_path in self.tifs:
            image = self.imagetypes.Image(image_path)
            array = self.imagetypes.Array(image_path)
            batch0 = self.Batch(image, array)
            self.assertEqual(len(batch0), 2)
            self.assertEqual(batch0.width,  image.width)
            self.assertEqual(batch0.height, image.height)
            self.assertEqual(batch0.width,  array.width)
            self.assertEqual(batch0.height, array.height)
            batch1 = self.Batch()
            batch1.append(self.imagetypes.Image(image_path))
            batch1.append(self.imagetypes.Array(image_path))
            self.assertEqual(len(batch0), len(batch1))
            batch1.append(imread.imread(image_path))
            self.assertFalse(batch0 == batch1)
    
    def test_batch_load_pvrs(self):
        for image_path in self.pvrs:
            image = self.imagetypes.Image(image_path)
            array = self.imagetypes.Array(image_path)
            batch0 = self.Batch(image, array)
            self.assertEqual(len(batch0), 2)
            self.assertEqual(batch0.width,  image.width)
            self.assertEqual(batch0.height, image.height)
            self.assertEqual(batch0.width,  array.width)
            self.assertEqual(batch0.height, array.height)
            batch1 = self.Batch()
            batch1.append(self.imagetypes.Image(image_path))
            batch1.append(self.imagetypes.Array(image_path))
            self.assertEqual(len(batch0), len(batch1))
            batch1.append(imread.imread(image_path))
            self.assertFalse(batch0 == batch1)
