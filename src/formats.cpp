/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/formats.hh>

#include <libimread/IO/apple.hh>
#include <libimread/IO/bmp.hh>
#include <libimread/IO/gif.hh>
#include <libimread/IO/hdf5.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/IO/lsm.hh>
#include <libimread/IO/png.hh>
#include <libimread/IO/ppm.hh>
#include <libimread/IO/pvrtc.hh>
#include <libimread/IO/tiff.hh>
#include <libimread/IO/webp.hh>
//#include <libimread/IO/xcassets.hh>

namespace im {
    
    namespace detail {
        
        bool ext(char const* format, char const* suffix) {
            return !std::strcmp(format, suffix);
        }
        
    } /* namespace detail */
    
    ImageFormat::unique_t get_format(char const* format) {
        using format_ptr = ImageFormat::unique_t;
        
        if (detail::ext(format, "png"))         { return format_ptr(new format::PNG);  }
        if (detail::ext(format, "jpg")  ||
            detail::ext(format, "jpeg"))        { return format_ptr(new format::JPG);  }
        if (detail::ext(format, "tif")  ||
            detail::ext(format, "tiff"))        { return format_ptr(new format::TIFF); }
        if (detail::ext(format, "pvr")  ||
            detail::ext(format, "pvrtc"))       { return format_ptr(new format::PVR);  }
        
        if (detail::ext(format, "hdf5") ||
            detail::ext(format, "h5")   ||
            detail::ext(format, "hdf"))         { return format_ptr(new format::HDF5); }
        
        if (detail::ext(format, "webp"))        { return format_ptr(new format::WebP); }
        if (detail::ext(format, "ppm"))         { return format_ptr(new format::PPM);  }
        if (detail::ext(format, "bmp"))         { return format_ptr(new format::BMP);  }
        if (detail::ext(format, "lsm"))         { return format_ptr(new format::LSM);  }
        if (detail::ext(format, "stk"))         { return format_ptr(new format::STK);  }
        if (detail::ext(format, "gif"))         { return format_ptr(new format::GIF);  }
        
        /// save JPEG2000 files with apple I/O
        if (detail::ext(format, "jp2")  ||
            detail::ext(format, "jpe2") ||
            detail::ext(format, "jpg2"))        { return format_ptr(new format::NS);   }
        
        if (detail::ext(format, "objc") ||
            detail::ext(format, "ns"))          { return format_ptr(new format::NS);   }
        
        imread_raise(FormatNotFound,
            "Format Error:",
            "\tFile format not found for suffix:", format);
    }
    
    char const* magic_format(byte_source* src) {
        if (format::JPG::match_format(src))     { return "jpg";  }
        if (format::PNG::match_format(src))     { return "png";  }
        if (format::BMP::match_format(src))     { return "bmp";  }
        if (format::GIF::match_format(src))     { return "gif";  }
        if (format::HDF5::match_format(src))    { return "hdf5"; }
        if (format::TIFF::match_format(src))    { return "tif";  }
        if (format::STK::match_format(src))     { return "tif";  }
        if (format::WebP::match_format(src))    { return "webp"; }
        if (format::PPM::match_format(src))     { return "ppm";  }
        // if (format::PVR::match_format(src))  { return "pvr";  }
        
        imread_raise(FormatNotFound,
            "Format Error:",
            "\tFile suffix not found for byte source");
    }
    
    ImageFormat::unique_t for_source(byte_source* source) {
        return get_format(magic_format(source));
    }
    

} /* namespace im */