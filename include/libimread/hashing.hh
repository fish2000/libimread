/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HASHING_HH_
#define LIBIMREAD_HASHING_HH_

#include <array>
#include <bitset>
#include <algorithm>
#include <functional>

#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/pixels.hh>

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
    
    namespace orig {
        
        int cmpint(const void *pa, const void *pb) noexcept;
        int cmpfloat(const void *pa, const void *pb) noexcept;
        float median(int *data, int n);
        float medianf(float *data, int n);
        char* bits_to_hexhash(int *bits, int nbits);
        
        void blockhash_quick(int bits, unsigned char *data,
                             int width, int height, int **hash);
        
        void blockhash(int bits, unsigned char *data,
                       int width, int height, int **hash);
    
    };
    
    using im::Image;
    using im::byte;
    
    namespace detail {
        
        template <std::size_t NN, typename Iterator>
        float median(Iterator begin, Iterator end) {
            /// middle for odd-length, and "upper-middle" for even length
            std::array<int, NN> local;
            std::copy(begin, end, local.begin());
            Iterator middle = local.begin() + (local.end() - local.begin()) / 2;
            
            /// This function runs in O(n) on average
            std::nth_element(local.begin(), middle, local.end());
            
            if ((local.end() - local.begin()) % 2 != 0) { // odd length
                return *middle;
            } else {
                /// even length -- the "lower middle" is the max of the lower half
                Iterator lower_middle = std::max_element(local.begin(), middle);
                return (*middle + *lower_middle) / 2.0f;
            }
        }
        
    };
    
    template <std::size_t N = 8>
    std::bitset<N*N> blockhash_quick(Image& image) {
        constexpr int        NN = N*N;
        constexpr int        NN2 = NN/2;
        constexpr int        NN4 = NN/4;
        int                  i, x, y, ix, iy;
        int                  ii, alpha, value;
        int                  block_width;
        int                  block_height;
        float                m[4];
        std::bitset<NN>      out;
        std::array<int, NN>  blocks;
        
        block_width = image.dim(0) / N;
        block_height = image.dim(1) / N;
        im::pix::accessor<byte> at = image.access();
        
        for (y = 0; y < N; y++) {
            for (x = 0; x < N; x++) {
                value = 0;
                for (iy = 0; iy < block_height; iy++) {
                    for (ix = 0; ix < block_width; ix++) {
                        /// calculate offset
                        value += at((x * block_width + ix), (y * block_height + iy), 0)[0] +
                                 at((x * block_width + ix), (y * block_height + iy), 1)[0] +
                                 at((x * block_width + ix), (y * block_height + iy), 2)[0];
                    }
                }
                blocks[y * N + x] = value;
            }
        }
        
        for (i = 0; i < 4; i++) {
            auto ppbegin = &blocks[i*NN4];
            auto ppend = ppbegin + NN4;
            m[i] = detail::median<NN4>(ppbegin, ppend);
        }
        
        for (i = 0; i < NN; i++) {
            if (  ((blocks[i] < m[0]) && (i < NN4))
               || ((blocks[i] < m[1]) && (i >= NN4) && (i < NN2))
               || ((blocks[i] < m[2]) && (i >= NN2) && (i < NN4+NN2))
               || ((blocks[i] < m[3]) && (i >= NN2+NN4))
               )
            {
              out[i] = false;
            } else {
              out[i] = true;
            }
        }
        
        return out;
    }
    
    template <std::size_t N = 8>
    std::bitset<N*N> blockhash(Image& image) {
        
    }
    
};


#endif /// LIBIMREAD_HASHING_HH_