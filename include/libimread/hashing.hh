/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HASHING_HH_
#define LIBIMREAD_HASHING_HH_

#include <bitset>
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
    using im::byte;
    
    template <std::size_t N = 8>
    std::bitset<N*N> blockhash_quick(Image& image) {
        int                 i, x, y, ix, iy;
        int                 ii, alpha, value;
        int                 block_width;
        int                 block_height;
        int                *blocks;
        float               m[4];
        std::bitset<N*N>    out;
        
        block_width = image.dim(0) / N;
        block_height = image.dim(1) / N;
        blocks = (int *)calloc(N * N, sizeof(int));
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
           m[i] = median(&blocks[i*N*N/4], N*N/4);
        }
        
        for (i = 0; i < N*N; i++) {
            if (  ((blocks[i] < m[0]) && (i < N*N/4))
               || ((blocks[i] < m[1]) && (i >= N*N/4) && (i < N*N/2))
               || ((blocks[i] < m[2]) && (i >= N*N/2) && (i < N*N/4+N*N/2))
               || ((blocks[i] < m[3]) && (i >= N*N/2+N*N/4))
               )
            {
              out[i] = false;
            } else {
              out[i] = true;
            }
        }
        
        free(blocks);
        return out;
    }
    
    template <std::size_t N = 8>
    std::bitset<N*N> blockhash(Image& image) {
        
    }
    
};


#endif /// LIBIMREAD_HASHING_HH_