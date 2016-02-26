/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>

namespace im {
    
    std::size_t ImageList::hash(std::size_t seed) const noexcept {
        return std::accumulate(content.begin(), content.end(),
                               seed, ImageList::rehasher_t());
    }
    
}

namespace std {
    
    template <>
    void swap(im::ImageList& p0, im::ImageList& p1) noexcept {
        p0.swap(p1);
    }
    
}; /* namespace std */
