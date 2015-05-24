/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FORMATS_HH_
#define LIBIMREAD_FORMATS_HH_

#include <memory>
#include <string>
#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/imageformat.hh>

namespace im {
    
    namespace detail {
        inline bool ext(const char *format, const char *suffix) {
            return !std::strcmp(format, suffix);
        }
    }
    
    std::unique_ptr<ImageFormat> get_format(const char*);
    std::unique_ptr<ImageFormat> for_filename(const char*);
    std::unique_ptr<ImageFormat> for_filename(std::string&);
    std::unique_ptr<ImageFormat> for_filename(const std::string&);
    const char *magic_format(byte_source*);

}

#endif /// LIBIMREAD_FORMATS_HH_
