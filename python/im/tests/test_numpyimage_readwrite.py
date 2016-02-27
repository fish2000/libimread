
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
    
    def test_load_image_jpg_write_blob_tif_readback_options_good(self):
        ''' Load JPG files, write TIF blobs with readback + meta-options (good) '''
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
    
    def test_load_image_jpg_write_blob_tif_readback_options_weird(self):
        ''' Load JPG files, write TIF blobs with readback + meta-options (weird) '''
        types = (float, int, long, memoryview, str)
        doggkeys = ['yodogg:%s' % yipe.__name__ for yipe in types]
        for image_path in self.jpgs:
            image = im.NumpyImage(image_path)
            self.assertIsNotNone(image)
            memview = memoryview('YO DOGG I HEARD YOU LIKE VIEWS OF MEMORY')
            data = image.write(as_blob=True, options={ 'format' : "tif", 'tiff:metadata'            : True,
                                                                         'metadata'                 : "YO DOGG",
                                                                         'tiff:software-signature'  : "I HEARD YOU LIKE METADATA",
                                                       
                                                       'yodogg:legal-suboptions'    : dict(yodogg="I heard you like",
                                                                                           dictsinyour="dicts, dogg"),
                                                       'yodogg:legal-subtuple'      : tuple(['yo', 'dogg', "it's",
                                                                                             "a", 'subtuple']),
                                                       'yodogg:legal-sublist'       : ['yo', 'same', 'shit', 'dogg' ],
                                                       'yodogg:legal-subset'        : set(['yo', 'dogg',
                                                                                           'yo', 'seriously', 'dogg']),
                                                       'yodogg:more-sublist-types'  : ["yo", 7007, 3.14, 98767576465234L, "DOGG"],
                                                       'yodogg:float'               : 2.71818,
                                                       'yodogg:int'                 : 666,
                                                       'yodogg:long'                : 6666875764L,
                                                       'yodogg:memoryview'          : memview,
                                                       'yodogg:str'                 : "SO WE PUT ALL SORTS OF SHIT IN HERE I DUNNO DOGG"
                                                     })
            self.assertIsNotNone(data)
            opts = image.write_opts
            
            for key in doggkeys:
                self.assertTrue(key in opts)
            
            # sets avoid false negs due to nondeterministic ordering
            self.assertEqual(
                set(opts['yodogg:legal-suboptions'].keys()),
                set(['yodogg', 'dictsinyour']))
            self.assertEqual(
                set(opts['yodogg:legal-suboptions'].values()),
                set(["I heard you like", "dicts, dogg"]))
            
            self.assertEqual(opts['yodogg:legal-subtuple'].index('yo'), 0)
            self.assertEqual(opts['yodogg:legal-subtuple'].index('dogg'), 1)
            self.assertEqual(opts['yodogg:legal-subtuple'].index('subtuple'), 4)
            self.assertEqual(opts['yodogg:legal-sublist'].index('yo'), 0)
            self.assertEqual(opts['yodogg:legal-sublist'].index('dogg'), 3)
            
            # 'yo' and 'dogg' repeat in 'legal-subset'
            self.assertEqual(len(opts['yodogg:legal-subset']), 3)
            self.assertTrue('yo' in opts['yodogg:legal-subset'])
            self.assertTrue('dogg' in opts['yodogg:legal-subset'])
            self.assertTrue('seriously' in opts['yodogg:legal-subset'])
            
            # the true test of this extemporaneous bullshit
            self.assertEqual(
                opts['yodogg:memoryview'].tobytes(),
                'YO DOGG I HEARD YOU LIKE VIEWS OF MEMORY')
            
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