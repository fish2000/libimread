// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/formats.hh>
#include <libimread/IO/apple.hh>
#include <libimread/IO/bmp.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/IO/lsm.hh>
#include <libimread/IO/png.hh>
#include <libimread/IO/tiff.hh>
#include <libimread/IO/pvrtc.hh>
#include <libimread/IO/webp.hh>

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
        
        if (check(format, "objc") || check(format, "ns")) { return format_ptr(new format::NS); }
        
        return format_ptr(nullptr);
    }
    
    std::unique_ptr<ImageFormat> for_filename(const char *cfilename) {
        return get_format(split_filename(cfilename));
    }
    std::unique_ptr<ImageFormat> for_filename(std::string &filename) {
        return get_format(split_filename(filename.c_str()));
    }
    std::unique_ptr<ImageFormat> for_filename(const std::string &filename) {
        return get_format(split_filename(filename.c_str()));
    }
    
    const char *magic_format(byte_source *src) {
        if (format::PNG::match_format(src)) { return "png"; }
        if (format::JPG::match_format(src)) { return "jpeg"; }
        return 0;
    }

}