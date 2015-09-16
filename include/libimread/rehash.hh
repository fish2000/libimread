/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_REHASH_HH_
#define LIBIMREAD_REHASH_HH_

namespace detail {
    
    template <typename T> inline
    void rehash(std::size_t& seed, const T& v) {
        /// also cribbed from boost,
        /// via http://stackoverflow.com/a/23860042/298171
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    
}

#endif /// LIBIMREAD_REHASH_HH_