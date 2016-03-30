/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FORMATS_HH_
#define LIBIMREAD_FORMATS_HH_

#include <memory>
#include <utility>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/imageformat.hh>

namespace im {
    
    ImageFormat::unique_t get_format(char const*);
    
    template <typename S> inline
    ImageFormat::unique_t for_filename(S&& s) {
        return get_format(
            filesystem::path::extension<S>(std::forward<S>(s)).c_str());
    }
    
    char const* magic_format(byte_source*);
    ImageFormat::unique_t for_source(byte_source*);
    
} /* namespace im */

#endif /// LIBIMREAD_FORMATS_HH_
