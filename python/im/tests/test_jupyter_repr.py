
from __future__ import print_function
from basecase import BaseCase

class JupyterReprTests(BaseCase):
    
    def test_jupyter_repr_jpeg(self):
        ''' Load some JPG and PNG files, write JPG blobs,
            compare to image._repr_jpeg_() '''
        for ImageType in self.imagetypes:
            
            for image_path in list(self.jpgs)[:10]:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                data = image.write(as_blob=True, options={ 'format' : "jpg" })
                self.assertIsNotNone(data)
                repr_jpeg = image._repr_jpeg_()
                self.assertEqual(data, repr_jpeg)
            
            for image_path in list(self.pngs)[:10]:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                data = image.write(as_blob=True, options={ 'format' : "jpg" })
                self.assertIsNotNone(data)
                repr_jpeg = image._repr_jpeg_()
                self.assertEqual(data, repr_jpeg)
    
    def test_jupyter_repr_png(self):
        ''' Load some JPG and PNG files, write PNG blobs,
            compare to image._repr_png_() '''
        for ImageType in self.imagetypes:
            
            for image_path in list(self.jpgs)[:10]:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                data = image.write(as_blob=True, options={ 'format' : "png" })
                self.assertIsNotNone(data)
                repr_png = image._repr_png_()
                self.assertEqual(data, repr_png)
            
            for image_path in list(self.pngs)[:10]:
                image = ImageType(image_path)
                self.assertIsNotNone(image)
                data = image.write(as_blob=True, options={ 'format' : "png" })
                self.assertIsNotNone(data)
                repr_png = image._repr_png_()
                self.assertEqual(data, repr_png)
