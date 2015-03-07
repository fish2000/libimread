// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/formats.hh>
#include <libimread/_apple.hh>
#include <libimread/_bmp.hh>
#include <libimread/_jpeg.hh>
#include <libimread/_lsm.hh>
#include <libimread/_png.hh>
#include <libimread/_tiff.hh>
#include <libimread/_pvrtc.hh>
#include <libimread/_webp.hh>

namespace im {
    
    namespace {
        inline bool check(const char *format, const char *suffix) {
            return !std::strcmp(format, suffix);
        }
    }
    
    std::unique_ptr<ImageFormat> get_format(const char *format) {
        using format_ptr = std::unique_ptr<ImageFormat>;
        
        if (check(format, "png")) { return format_ptr(new format::PNG); }
        if (check(format, "jpg") || check(format, "jpeg")) { return format_ptr(new format::JPG); }
        if (check(format, "tif") || check(format, "tiff")) { return format_ptr(new format::TIFF); }
        if (check(format, "pvr") || check(format, "pvrtc")) { return format_ptr(new format::PVR); }
        if (check(format, "webp")) { return format_ptr(new format::WebP); }
        if (check(format, "bmp")) { return format_ptr(new format::BMP); }
        if (check(format, "lsm")) { return format_ptr(new format::LSM); }
        if (check(format, "stk")) { return format_ptr(new format::STK); }
        
        return format_ptr(nullptr);
    }
    
    std::unique_ptr<ImageFormat> format_for_filename(const char *cfilename) {
        return get_format(split_filename(cfilename));
    }
    std::unique_ptr<ImageFormat> format_for_filename(std::string &filename) {
        return get_format(split_filename(filename.c_str()));
    }
    std::unique_ptr<ImageFormat> format_for_filename(const std::string &filename) {
        return get_format(split_filename(filename.c_str()));
    }
    
    const char *magic_format(byte_source *src) {
        if (format::PNG::match_format(src)) { return "png"; }
        if (format::JPG::match_format(src)) { return "jpeg"; }
        return 0;
    }

}