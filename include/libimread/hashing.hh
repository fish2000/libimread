/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HASHING_HH_
#define LIBIMREAD_HASHING_HH_

#include <bitset>
#include <memory>
#include <functional>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>

namespace detail {
    
    template <typename T> inline
    void rehash(std::size_t& seed, const T& v) {
        /// also cribbed from boost,
        /// via http://stackoverflow.com/a/23860042/298171
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    
}

namespace blockhash {
    
    int cmpint(const void *pa, const void *pb) noexcept;
    int cmpfloat(const void *pa, const void *pb) noexcept;
    float median(int *data, int n);
    float medianf(float *data, int n);
    char* bits_to_hexhash(int *bits, int nbits);
    
    namespace orig {
    
        void blockhash_quick(int bits, unsigned char *data,
                             int width, int height, int **hash);
        
        void blockhash(int bits, unsigned char *data,
                       int width, int height, int **hash);
    
    };
    
    using im::Image;
    
    template <std::size_t N = 8>
    std::bitset<N*N> blockhash_quick(std::unique_ptr<Image> image);
    
    template <std::size_t N = 8>
    std::bitset<N*N> blockhash(std::unique_ptr<Image> image);
    
};


#endif /// LIBIMREAD_HASHING_HH_