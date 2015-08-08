/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>

#include <libimread/formats.hh>
#include <libimread/IO/apple.hh>
#include <libimread/IO/bmp.hh>
#include <libimread/IO/gif.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/IO/lsm.hh>
#include <libimread/IO/png.hh>
#include <libimread/IO/ppm.hh>
#include <libimread/IO/pvrtc.hh>
#include <libimread/IO/tiff.hh>
#include <libimread/IO/webp.hh>
//#include <libimread/IO/xcassets.hh>

namespace im {
    
    std::unique_ptr<ImageFormat> get_format(const char *format) {
        using format_ptr = std::unique_ptr<ImageFormat>;
        
        if (detail::ext(format, "png")) { return format_ptr(new format::PNG); }
        if (detail::ext(format, "jpg") || detail::ext(format, "jpeg")) { return format_ptr(new format::JPG); }
        if (detail::ext(format, "tif") || detail::ext(format, "tiff")) { return format_ptr(new format::TIFF); }
        if (detail::ext(format, "pvr") || detail::ext(format, "pvrtc")) { return format_ptr(new format::PVR); }
        if (detail::ext(format, "webp")) { return format_ptr(new format::WebP); }
        if (detail::ext(format, "ppm")) { return format_ptr(new format::PPM); }
        if (detail::ext(format, "bmp")) { return format_ptr(new format::BMP); }
        if (detail::ext(format, "lsm")) { return format_ptr(new format::LSM); }
        if (detail::ext(format, "stk")) { return format_ptr(new format::STK); }
        if (detail::ext(format, "gif")) { return format_ptr(new format::GIF); }
        
        if (detail::ext(format, "objc") || detail::ext(format, "ns")) { return format_ptr(new format::NS); }
        
        imread_raise(FormatNotFound, "Format Error:",
            FF("\tFile format not found for suffix %s", format));
    }
    
    const char *magic_format(byte_source *src) {
        if (format::BMP::match_format(src)) { return "bmp"; }
        if (format::GIF::match_format(src)) { return "gif"; }
        if (format::JPG::match_format(src)) { return "jpg"; }
        if (format::PNG::match_format(src)) { return "png"; }
        if (format::TIFF::match_format(src)) { return "tif"; }
        return 0;
    }

}