
from __future__ import print_function
from basecase import BaseCase

import os
import im
import json

class ArrayReadWriteTests(BaseCase):
    
    def test_image_check(self):
        # ''' Re-using `test_load_array_jpg_write_blob_jpg` here '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            self.assertTrue(im.Array.check(image))
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
            self.assertFalse(im.Array.check(data))
    
    def test_load_array_jpg_write_blob_jpg(self):
        ''' Load JPG files, write JPG blobs '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
    
    def test_load_array_png_write_blob_jpg(self):
        ''' Load PNG files, write JPG blobs '''
        for image_path in self.pngs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
    
    def test_load_array_jpg_write_blob_png(self):
        ''' Load JPG files, write PNG blobs '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
    
    def test_load_array_png_write_blob_png(self):
        ''' Load PNG files, write PNG blobs '''
        for image_path in self.pngs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
    
    def test_load_array_jpg_write_blob_tif(self):
        ''' Load JPG files, write TIF blobs '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "tif" })
            self.assertIsNotNone(data)
    
    def test_load_array_jpg_write_blob_png_readback(self):
        ''' Load JPG files, write PNG blobs with readback '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
            image2 = im.Array(data, is_blob=True)
            self.assertIsNotNone(image2)
            self.assertEqual(image.shape, image2.shape)
    
    def test_load_array_png_write_blob_png_readback(self):
        ''' Load PNG files, write PNG blobs with readback '''
        for image_path in self.pngs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "png" })
            self.assertIsNotNone(data)
            image2 = im.Array(data, is_blob=True)
            self.assertIsNotNone(image2)
            # self.assertEqual(image.shape[:2], image2.shape[:2])
            self.assertEqual(image.width, image2.width)
            self.assertEqual(image.height, image2.height)
    
    def test_load_array_jpg_write_blob_jpg_readback(self):
        ''' Load JPG files, write PNG blobs with readback '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
            image2 = im.Array(data, is_blob=True)
            self.assertIsNotNone(image2)
            self.assertEqual(image.shape, image2.shape)
    
    def test_load_array_png_write_blob_jpg_readback(self):
        ''' Load PNG files, write PNG blobs with readback '''
        for image_path in self.pngs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "jpg" })
            self.assertIsNotNone(data)
            image2 = im.Array(data, is_blob=True)
            self.assertIsNotNone(image2)
            # self.assertEqual(image.shape[:2], image2.shape[:2])
            self.assertEqual(image.width, image2.width)
            self.assertEqual(image.height, image2.height)
    
    def test_load_array_jpg_write_blob_tif_readback(self):
        ''' Load JPG files, write TIF blobs with readback '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "tif" })
            self.assertIsNotNone(data)
            image2 = im.Array(data, is_blob=True)
            self.assertIsNotNone(image2)
            # self.assertEqual(image.shape[:2], image2.shape[:2])
            self.assertEqual(image.width, image2.width)
            self.assertEqual(image.height, image2.height)
    
    def test_load_array_jpg_write_blob_tif_readback_options_good(self):
        ''' Load JPG files, write TIF blobs with readback + meta-options (good) '''
        for image_path in self.jpgs:
            image = im.Array(image_path)
            self.assertIsNotNone(image)
            data = image.write(as_blob=True, options={ 'format' : "tif", 'tiff:metadata'            : True,
                                                                         'metadata'                 : "YO DOGG",
                                                                         'tiff:software-signature'  : "I HEARD YOU LIKE METADATA" })
            self.assertIsNotNone(data)
            image2 = im.Array(data, is_blob=True)
            self.assertIsNotNone(image2)
            # self.assertEqual(image.shape[:2], image2.shape[:2])
            self.assertEqual(image.width, image2.width)
            self.assertEqual(image.height, image2.height)
    
    def test_load_array_jpg_write_blob_tif_readback_options_weird(self):
        ''' Load JPG files, write TIF blobs with readback + meta-options (weird) '''
        types = (float, int, long, memoryview, str)
        doggkeys = ['yodogg:%s' % yipe.__name__ for yipe in types]
        for image_path in self.jpgs:
            image = im.Array(image_path)
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
            opts_json = image.format_write_opts()
            self.assertIsNotNone(data)
            self.assertIsNotNone(opts_json)
            self.assertEqual(opts_json, image.format_write_opts())
            opts_tempfile = image.dump_write_opts(tempfile=True)
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
            
            image2 = im.Array(data, is_blob=True)
            self.assertIsNotNone(image2)
            # self.assertEqual(image.shape[:2], image2.shape[:2])
            self.assertEqual(image.width, image2.width)
            self.assertEqual(image.height, image2.height)
            
            with open(opts_tempfile, "rb") as fh:
                opts_reconstituted = json.load(fh)
            self.assertIsNotNone(opts_reconstituted)
            self.assertDictEqual(opts_reconstituted, json.loads(opts_json))
            
            os.remove(opts_tempfile)
    