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
    
    std::unique_ptr<ImageFormat> get_format(const char*);
    
    template <typename S> inline
    std::unique_ptr<ImageFormat> for_filename(S&& s) {
        return get_format(
            filesystem::path::extension<S>(std::forward<S>(s)).c_str());
    }
    
    const char* magic_format(byte_source*);
    std::unique_ptr<ImageFormat> for_source(byte_source*);
    
} /* namespace im */

#endif /// LIBIMREAD_FORMATS_HH_
