/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/image.hh>

namespace im {}

namespace std {
    
    template <>
    void swap(im::ImageList& p0, im::ImageList& p1) noexcept {
        p0.swap(p1);
    }
    
}; /* namespace std */
