/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/formats.hh>

#include <libimread/IO/all.hh>

namespace im {
    
    namespace detail {
        
        bool ext(char const* format, char const* suffix) {
            return !std::strcmp(format, suffix);
        }
        
    } /* namespace detail */
    
    ImageFormat::unique_t get_format(char const* format) {
        using format_ptr = ImageFormat::unique_t;
        
        if (detail::ext(format, "jpg")   ||
            detail::ext(format, "jpe")   ||
            detail::ext(format, "jpeg"))        { return format_ptr(new format::JPG);  }
        
        if (detail::ext(format, "png"))         { return format_ptr(new format::PNG);  }
        if (detail::ext(format, "gif"))         { return format_ptr(new format::GIF);  }
        if (detail::ext(format, "tif")   ||
            detail::ext(format, "tiff"))        { return format_ptr(new format::TIFF); }
        
        if (detail::ext(format, "hdf5")  ||
            detail::ext(format, "h5")    ||
            detail::ext(format, "hdf"))         { return format_ptr(new format::HDF5); }
        
        if (detail::ext(format, "pvr")   ||
            detail::ext(format, "pvrtc"))       { return format_ptr(new format::PVR);  }
        
        if (detail::ext(format, "webp"))        { return format_ptr(new format::WebP); }
        if (detail::ext(format, "bmp"))         { return format_ptr(new format::BMP);  }
        if (detail::ext(format, "ppm"))         { return format_ptr(new format::PPM);  }
        if (detail::ext(format, "stk"))         { return format_ptr(new format::STK);  }
        if (detail::ext(format, "lsm"))         { return format_ptr(new format::LSM);  }
        
        imread_raise(FormatNotFound,
            "Format Error:",
            "\tFile format not found for suffix:", format);
    }
    
    char const* magic_format(byte_source* source) {
        if  (format::JPG::match_format(source)) { return "jpg";  }
        if  (format::PNG::match_format(source)) { return "png";  }
        if  (format::GIF::match_format(source)) { return "gif";  }
        if (format::TIFF::match_format(source)) { return "tif";  }
        if (format::HDF5::match_format(source)) { return "hdf5"; }
        if (format::WebP::match_format(source)) { return "webp"; }
        if  (format::PVR::match_format(source)) { return "pvr";  }
        if  (format::STK::match_format(source)) { return "stk";  }
        if  (format::BMP::match_format(source)) { return "bmp";  }
        if  (format::PPM::match_format(source)) { return "ppm";  }
        
        imread_raise(FormatNotFound,
            "Format Error:",
            "\tFile suffix not found for byte source");
    }
    
    ImageFormat::unique_t for_source(byte_source* source) {
        return im::get_format(
               im::magic_format(source));
    }
    
} /* namespace im */