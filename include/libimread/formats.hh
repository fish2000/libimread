/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FORMATS_HH_
#define LIBIMREAD_FORMATS_HH_

#include <cctype>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/seekable.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    using filesystem::path;
    
    ImageFormat::unique_t get_format(char const*);
    char const*         magic_format(byte_source*);
    ImageFormat::unique_t for_source(byte_source*);
    
    template <typename S> inline
    ImageFormat::unique_t for_filename(S&& s) {
        /// get format name file extension
        std::string format = path::extension<S>(
                                std::forward<S>(s));
        
        /// convert name to lowercase
        std::transform(format.begin(),
                       format.end(),
                       format.begin(), [](unsigned char c) { return std::tolower(c); });
        
        /// return format type for name
        return im::get_format(format.c_str());
    }
    
} /* namespace im */

#endif /// LIBIMREAD_FORMATS_HH_
