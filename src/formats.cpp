// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <libimread/formats.h>
#include <libimread/_bmp.h>
#include <libimread/_jpeg.h>
#include <libimread/_lsm.h>
#include <libimread/_png.h>
#include <libimread/_tiff.h>
#include <libimread/_pvrtc.h>
#include <libimread/_webp.h>

namespace im {

    std::unique_ptr<ImageFormat> get_format(const char* format) {
        using std::strcmp;
        if (!strcmp(format, "png")) return std::unique_ptr<ImageFormat>(new PNGFormat);
        if (!strcmp(format, "jpeg") || !strcmp(format, "jpg")) return std::unique_ptr<ImageFormat>(new JPEGFormat);
        if (!strcmp(format, "lsm")) return std::unique_ptr<ImageFormat>(new LSMFormat);
        if (!strcmp(format, "tiff") || !strcmp(format, "tif")) return std::unique_ptr<ImageFormat>(new TIFFFormat);
        
        if (!strcmp(format, "pvr")) return std::unique_ptr<ImageFormat>(new PVRTCFormat);
        if (!strcmp(format, "pvrtc")) return std::unique_ptr<ImageFormat>(new PVRTCFormat);
        
        if (!strcmp(format, "stk")) return std::unique_ptr<ImageFormat>(new STKFormat);
        if (!strcmp(format, "bmp")) return std::unique_ptr<ImageFormat>(new BMPFormat);
        if (!strcmp(format, "webp")) return std::unique_ptr<ImageFormat>(new WebPFormat);
        return std::unique_ptr<ImageFormat>(nullptr);
    }
    
    const char* magic_format(byte_source* src) {
        if (PNGFormat::match_format(src)) return "png";
        if (JPEGFormat::match_format(src)) return "jpeg";
        return 0;
    }

}