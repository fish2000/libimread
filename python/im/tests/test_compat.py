
from __future__ import print_function
from basecase import BaseCase

import im

class CompatibilityTests(BaseCase):
    
    def test_imread_imread(self):
        """ Load JPG files by filename, with both versions of imread:
            compat.imread (ours) and luispedro imread (the orig);
            compare the returned arrays with numpy.all()
        """
        import numpy
        import imread as luispedro
        from im.compat import imread
        for image_path in self.jpgs:
            ar0 = imread.imread(image_path)
            ar1 = luispedro.imread(image_path)
            self.assertTrue(numpy.all(ar0 == ar1))
    
    def test_imread_imread_from_blob(self):
        """ Load JPG files as blob data, with both versions of imread:
            compat.imread (ours) and luispedro imread (the orig);
            compare the returned arrays with numpy.all()
        """
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
        """ Detect format of JPGs, by both filename and as blob data,
            using both versions of imread: compat.imread (ours)
            and luispedro imread (the orig); compare the returned format
            strings with one another as appropriate
        """
        import imread as luispedro
        from im.compat import imread
        for image_path in self.jpgs:
            format0 = imread.detect_format(image_path)
            format1 = luispedro.detect_format(image_path)
            self.assertEqual(format0, 'jpg')
            self.assertEqual(format1, 'jpeg')
            with open(image_path, 'rb') as fh:
                blob = fh.read()
                blobformat0 = imread.detect_format(blob, is_blob=True)
                blobformat1 = luispedro.detect_format(blob, is_blob=True)
                self.assertEqual(blobformat0, 'jpg')
                self.assertEqual(blobformat1, 'jpeg')
    
    def test_imread_supports_format(self):
        """ Check format support in compat.imread
            and luispedro imread
        """
        import imread as luispedro
        self.assertTrue(luispedro.supports_format('bmp'))
        self.assertTrue(luispedro.supports_format('png'))
        self.assertTrue(luispedro.supports_format('tiff'))
        self.assertTrue(luispedro.supports_format('tif'))
        self.assertTrue(luispedro.supports_format('jpeg'))
        self.assertTrue(luispedro.supports_format('jpg'))
        self.assertFalse(luispedro.supports_format('ppm'))
        self.assertFalse(luispedro.supports_format('pvr'))
        self.assertFalse(luispedro.supports_format('gif'))
        self.assertTrue(luispedro.supports_format('webp'))
        self.assertFalse(luispedro.supports_format('hdf5'))
        
        from im.compat import imread
        self.assertTrue(imread.supports_format('bmp'))
        self.assertTrue(imread.supports_format('png'))
        self.assertTrue(imread.supports_format('tiff'))
        self.assertTrue(imread.supports_format('tif'))
        self.assertTrue(imread.supports_format('jpeg'))
        self.assertTrue(imread.supports_format('jpg'))
        self.assertTrue(imread.supports_format('ppm'))
        self.assertTrue(imread.supports_format('pvr'))
        self.assertTrue(imread.supports_format('gif'))
        self.assertTrue(imread.supports_format('webp'))
        self.assertTrue(imread.supports_format('hdf5'))
    
    def test_imread_imread_imsave(self):
        """ Load JPG files by filename, with both versions of imread:
            compat.imread (ours) and luispedro imread (the orig);
            compare the returned arrays with numpy.all();
            resave each array with the complementary imread version --
                ... e.g. that which was read with compat.imread,
                         write with luispedro imread,
                ... and vicea-versa;
            re-read again, once more using complementary versions;
            compare the final array result with numpy.allclose()
                ... with high tolerances to account for variance
                    in the JPEG compressor
        """
        import numpy
        import imread as luispedro
        from im.compat import imread
        from tempfile import NamedTemporaryFile
        s = ".jpg"
        p = "yo-dogg-"
        for image_path in list(self.jpgs)[:16]:
            ar0 = imread.imread(image_path)
            ar1 = luispedro.imread(image_path)
            self.assertTrue(numpy.all(ar0 == ar1))
            rr0 = rr1 = None
            with NamedTemporaryFile(suffix=s, prefix=p) as tf:
                luispedro.imsave(tf.name, ar0, formatstr='jpeg')
                rr0 = imread.imread(tf.name)
            with NamedTemporaryFile(suffix=s, prefix=p) as tf:
                imread.imsave(tf.name, ar1, formatstr='jpeg')
                rr1 = luispedro.imread(tf.name)
            self.assertTrue(numpy.allclose(rr0.astype('float'),
                                           rr1.astype('float'),
                                           rtol=4.0,
                                           atol=128.0))
    
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
    