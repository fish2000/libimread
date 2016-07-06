
from __future__ import print_function
from basecase import BaseCase

import im

class BatchTests(BaseCase):
    
    def test_batch_load_jpgs(self):
        for image_path in self.jpgs:
            image = self.imagetypes.Image(image_path)
            array = self.imagetypes.Array(image_path)
            batch0 = im.Batch(image, array)
            self.assertEqual(len(batch0), 2)
            self.assertEqual(batch0.width,  image.width)
            self.assertEqual(batch0.height, image.height)
            self.assertEqual(batch0.width,  array.width)
            self.assertEqual(batch0.height, array.height)
            # batch1 = im.Batch()
            # batch1.append(self.imagetypes.Image(image_path))
            # batch1.append(self.imagetypes.Array(image_path))
            # self.assertEqual(len(batch0), len(batch1))
            # batch1.append(image)
            # self.assertFalse(batch0 == batch1)
    
    def test_batch_load_pngs(self):
        for image_path in self.pngs:
            image = self.imagetypes.Image(image_path)
            array = self.imagetypes.Array(image_path)
            batch0 = self.Batch(image, array)
            self.assertEqual(batch0.width,  image.width)
            self.assertEqual(batch0.height, image.height)
            self.assertEqual(batch0.width,  array.width)
            self.assertEqual(batch0.height, array.height)
            # batch1 = self.Batch()
            # batch1.append(self.imagetypes.Image(image_path))
            # batch1.append(self.imagetypes.Array(image_path))
            # self.assertEqual(len(batch0), len(batch1))
            # batch1.append(image)
            # self.assertFalse(batch0 == batch1)
    
    def test_batch_load_tifs(self):
        for image_path in self.tifs:
            image = self.imagetypes.Image(image_path)
            array = self.imagetypes.Array(image_path)
            batch0 = self.Batch(image, array)
            self.assertEqual(batch0.width,  image.width)
            self.assertEqual(batch0.height, image.height)
            self.assertEqual(batch0.width,  array.width)
            self.assertEqual(batch0.height, array.height)
            # batch1 = self.Batch()
            # batch1.append(self.imagetypes.Image(image_path))
            # batch1.append(self.imagetypes.Array(image_path))
            # self.assertEqual(len(batch0), len(batch1))
            # batch1.append(image)
            # self.assertFalse(batch0 == batch1)
    
    def test_batch_load_pvrs(self):
        for image_path in self.pvrs:
            image = self.imagetypes.Image(image_path)
            array = self.imagetypes.Array(image_path)
            batch0 = self.Batch(image, array)
            self.assertEqual(batch0.width,  image.width)
            self.assertEqual(batch0.height, image.height)
            self.assertEqual(batch0.width,  array.width)
            self.assertEqual(batch0.height, array.height)
            # batch1 = self.Batch()
            # batch1.append(self.imagetypes.Image(image_path))
            # batch1.append(self.imagetypes.Array(image_path))
            # self.assertEqual(len(batch0), len(batch1))
            # batch1.append(image)
            # self.assertFalse(batch0 == batch1)
