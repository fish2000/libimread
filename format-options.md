
SYMBOLIZE.
==========

EXISTING SYMBOLS
-----------------------------

* `jpeg:quality`
* `png:compression_level`
* `tiff:compress`
* `tiff:horizontal-predictor`
* `metadata`
* `tiff:XResolution` (?)
* `tiff:YResolution` (?)
* `tiff:XResolutionUnit` (?)

        JPEGFormat::write() -> (int)            'jpeg:quality'              ;
        PNGFormat::write()  -> (int)            'png:compression_level'     ;
        TIFFFormat::write() -> (bool)           'tiff:compress'             ;
        TIFFFormat::write() -> (bool)           'tiff:horizontal-predictor' ;  /// depends on tiff:compress
        TIFFFormat::write() -> (const char *)   'metadata'                  ;
        TIFFFormat::write() -> (int|float)      'tiff:XResolution'          ;  /// can be either type (?!)
        TIFFFormat::write() -> (int|float)      'tiff:YResolution'          ;  /// can also be either type (srsly WTF)
        TIFFFormat::write() -> (int)            'tiff:XResolutionUnit'      ;  /// int value likely maps to TIFF library's
                                                                               /// constant or LUT definitions
                                                                               /// ... also value goes into tiff tag
                                                                               /// 'RESOLUTIONUNIT' (used for both X and Y)
                                                                               /// despite being called XResolutionUnit
                                                                               /// (WOW WHAT IN FUCK MAN, GOD)

SYMBOLS YET TO BE WRITTEN
---------------------------------------------

¶ N.B. void* really just means "TBD" here, and will be some other ptr/ref type

* `png:ios-premultiply`
* `md:icc`
* `md:xmp-data` (blech)
* `md:xmp-sidecar` (also blech)
* `md:exif` (yeccch)
* `md:thumbnail` (OH PLEEZE)

        PNGFormat::write()   -> (bool)                 'png:ios-premultiply'      ;  /// write an iOS-ready PNG file if true
        ImageFormat::write() -> (std::string|void*)    'md:icc'                  ;  /// metadata -> ICC profile (JPEG, PNG-24, PNG-8(??), TIFF, [PSD])
        ImageFormat::write() -> (std::string|void*)    'md:xmp-data'             ;  /// metadata -> XMP catalog data
        ImageFormat::write() -> (bool)                 'md:xmp-sidecar'          ;  /// metadata -> write XMP data to 'sidecar' .xmp file (w/o embedding) if true
        ImageFormat::write() -> (std::string)          'md:exif'                 ;  /// metadata -> EXIF tag data
        ImageFormat::write() -> (std::string)          'md:thumbnail'            ;  /// metadata -> thumbnail image (for EXIF, QuickLook, etc)

<hr>

START TIFF SHIT
==============

    if (get_optional_bool(opts, "tiff:compress", true)) {
        TIFFSetField(t.tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
        // For 8 bit images, prediction defaults to false; for 16 bit images,
        // it defaults to true. This is because compression of raw 16 bit
        // images is often counter-productive without this flag. See the
        // discusssion at http://www.asmail.be/msg0055176395.html
        const bool prediction_default = input.nbits() != 8;
        if (get_optional_bool(opts, "tiff:horizontal-predictor", prediction_default)) {
            TIFFSetField(t.tif, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
            if (!copy_data) {
                bufdata.resize(input.dim(1) * input.nbytes());
                bufp = &bufdata[0];
                copy_data = true;
            }
        }
    }
    
    TIFFSetField(t.tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    const char *meta = get_optional_cstring(opts, "metadata");
    if (meta) {
        TIFFSetField(t.tif, TIFFTAG_IMAGEDESCRIPTION, meta);
    }
    Options::const_iterator x_iter = opts.find("tiff:XResolution");
    if (x_iter != opts.end()) {
        double d;
        int i;
        float value;
        if (x_iter->second.get_int(i)) {
            value = i;
        } else if (x_iter->second.get_double(d)) {
            value = d;
        } else {
            throw WriteOptionsError("im::TIFFFormat::write(): XResolution must be an integer or floating point value.");
        }
        TIFFSetField(t.tif, TIFFTAG_XRESOLUTION, value);
    }
    
    Options::const_iterator y_iter = opts.find("tiff:YResolution");
    if (y_iter != opts.end()) {
        double d;
        int i;
        float value;
        if (y_iter->second.get_int(i)) {
            value = i;
        } else if (y_iter->second.get_double(d)) {
            value = d;
        } else {
            throw WriteOptionsError("im::TIFFFormat::write(): YResolution must be an integer or floating point value.");
        }
    
        TIFFSetField(t.tif, TIFFTAG_YRESOLUTION, value);
    }
    
    const uint16_t resolution_unit = get_optional_int(opts, "tiff:XResolutionUnit", uint16_t(-1));
    if (resolution_unit != uint16_t(-1)) {
        TIFFSetField(t.tif, TIFFTAG_RESOLUTIONUNIT, resolution_unit);
    }
    
END TIFF SHIT
============

<hr>

START PNG SHIT
==============

        int compression_level = get_optional_int(opts, "png:compression_level", -1);
        if (compression_level != -1) {
            png_set_compression_level(p.png_ptr, compression_level);
        }
        
        Options::const_iterator qiter = opts.find("jpeg:quality");
        if (qiter != opts.end()) {
            int quality;
            if (qiter->second.get_int(quality)) {
                if (quality > 100) { quality = 100; }
                if (quality < 0) { quality = 0; }
                jpeg_set_quality(&compressor.info, quality, FALSE);
            } else {
                throw WriteOptionsError(
                    "im::JPEGFormat::write(): jpeg:quality must be an integer"
                );
            }
        }


END PNG SHIT
============

