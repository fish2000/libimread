
from __future__ import print_function
from basecase import BaseCase

import base64

class JupyterReprTests(BaseCase):
    
    def test_jupyter_repr_jpeg(self):
        ''' Load some JPG and PNG files, write JPG blobs,
            compare to image._repr_jpeg_() '''
        for ImageType in self.imagetypes:
            
            for image_path in list(self.jpgs | self.pngs)[:20]:
                image = ImageType(image_path)
                # self.assertIsNotNone(image)
                data = image.write(as_blob=True, options={ 'format' : "jpg" })
                repr_jpeg = image._repr_jpeg_()
                self.assertEqual(data, repr_jpeg)
    
    def test_jupyter_repr_png(self):
        ''' Load some JPG and PNG files, write PNG blobs,
            compare to image._repr_png_() '''
        for ImageType in self.imagetypes:
            
            for image_path in list(self.jpgs | self.pngs)[:10]:
                image = ImageType(image_path)
                # self.assertIsNotNone(image)
                data = image.write(as_blob=True, options={ 'format' : "png" })
                repr_png = image._repr_png_()
                self.assertEqual(data, repr_png)
    
    def test_jupyter_repr_html(self):
        ''' Load some JPG and PNG files, write JPG blobs,
            encode blobs to base64 and prepare HTML tags,
            compare to image._repr_html_() '''
        for ImageType in self.imagetypes:
            
            for image_path in list(self.jpgs | self.pngs)[:20]:
                image = ImageType(image_path)
                # self.assertIsNotNone(image)
                data = image.write(as_blob=True, options={ 'format' : "jpg" })
                tag = "<img src='data:image/jpeg;base64,%s'>" % base64.b64encode(data)
                repr_html = image._repr_html_()
                self.assertEqual(tag, repr_html)
