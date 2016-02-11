/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HASHING_HH_
#define LIBIMREAD_HASHING_HH_

#include <cstring>
#include <cmath>
#include <array>
#include <bitset>
#include <algorithm>
#include <functional>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/ext/butteraugli.hh>
#include <libimread/image.hh>
#include <libimread/pixels.hh>

using im::Image;
using im::byte;

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
    
    namespace detail {
        
        template <std::size_t NN, typename Iterator>
        float median(Iterator begin, Iterator end) {
            using Type = typename std::remove_pointer_t<std::decay_t<Iterator>>;
            
            /// middle for odd-length, and "upper-middle" for even length
            std::array<Type, NN> local;
            std::copy(begin, end, local.begin());
            auto diff = local.end() - local.begin();
            auto middle = local.begin() + diff / 2;
            
            /// This function runs in O(n) on average
            std::nth_element(local.begin(), middle, local.end());
            
            if (diff % 2 != 0) {
                /// odd length
                return *middle;
            } else {
                /// even length -- the "lower middle" is the max of the lower half
                auto lower_middle = std::max_element(local.begin(), middle);
                return (*middle + *lower_middle) / 2.0f;
            }
        }
        
        template <std::size_t N> inline
        std::string hexify(std::bitset<N> bits) {
            const std::string   prefix("0x");
            constexpr int       len = N / 4;
            unsigned int        i, j, b, t, tmp;
            char                stmp[2];
            std::string         out(len, ' ');
            
            for (i = 0; i < len; i++) {
                tmp = 0;
                for (j = 0; j < 4; j++) {
                    b = i * 4 + j;
                    t = uint8_t(bits[b]);
                    tmp |= (t << 3 >> j);
                }
                std::sprintf(stmp, "%1x", tmp);
                out[i] = stmp[0];
            }
            
            return prefix + out;
        }
        
    };
    
    template <std::size_t N = 8>
    std::bitset<N*N> blockhash_quick(Image& image) {
        constexpr int           NN = N*N;
        constexpr int           NN2 = NN/2;
        constexpr int           NN4 = NN/4;
        const int               block_width = image.dim(0) / N;
        const int               block_height = image.dim(1) / N;
        int                     i, x, y, ix, iy, value;
        float                   m[4];
        std::bitset<NN>         out;
        std::array<int, NN>     blocks;
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
        constexpr int               NN = N*N;
        constexpr int               NN2 = NN/2;
        constexpr int               NN4 = NN/4;
        const float                 block_width = (float) image.dim(0) / (float) N;
        const float                 block_height = (float) image.dim(1) / (float) N;
        const int                   width = image.dim(0);
        const int                   height = image.dim(1);
        float                       m[4];
        float                       y_frac, y_int;
        float                       x_frac, x_int;
        float                       x_mod, y_mod, value;
        float                       weight_top, weight_bottom, weight_left, weight_right;
        int                         block_top, block_bottom, block_left, block_right;
        int                         i, x, y;
        std::bitset<NN>             out;
        std::array<float, NN>       blocks;
        im::pix::accessor<byte>     at = image.access();
        
        if (width % N == 0 && height % N == 0) {
            return blockhash_quick<N>(image);
        }
        
        for (y = 0; y < height; y++) {
            y_mod = std::fmod(y + 1, block_height);
            y_frac = std::modf(y_mod, &y_int);
            
            weight_top = (1 - y_frac);
            weight_bottom = y_frac;
            
            /// y_int will be 0 on bottom/right borders and on block boundaries
            if (y_int > 0 || (y + 1) == height) {
                block_top = block_bottom = (int)std::floor((float) y / block_height);
            } else {
                block_top = (int)std::floor((float) y / block_height);
                block_bottom = (int)std::ceil((float) y / block_height);
            }
            
            for (x = 0; x < width; x++) {
                x_mod = std::fmod(x + 1, block_width);
                x_frac = std::modf(x_mod, &x_int);
                weight_left = (1 - x_frac);
                weight_right = x_frac;
                
                /// x_int will be 0 on bottom/right borders and on block boundaries
                if (x_int > 0 || (x + 1) == width) {
                    block_left = block_right = (int)std::floor((float) x / block_width);
                } else {
                    block_left = (int)std::floor((float) x / block_width);
                    block_right = (int)std::ceil((float) x / block_width);
                }
                
                /// get value at coords
                value = (float) at(x, y, 0)[0] +
                        (float) at(x, y, 1)[0] +
                        (float) at(x, y, 2)[0];
                
                /// add weighted pixel value to relevant blocks
                blocks[block_top * N + block_left] += value * weight_top * weight_left;
                blocks[block_top * N + block_right] += value * weight_top * weight_right;
                blocks[block_bottom * N + block_left] += value * weight_bottom * weight_left;
                blocks[block_bottom * N + block_right] += value * weight_bottom * weight_right;
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
    
}; /* namespace blockhash */

namespace std {
    
    /// std::hash specialization for im::Image
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<im::Image> {
        
        typedef im::Image argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type& image) const {
            auto bithash = blockhash::blockhash(image);
            return static_cast<result_type>(bithash.to_ullong());
        }
        
    };
    
}; /* namespace std */

namespace butteraugli {
    
    using im::Image;
    using pixel_t = double;
    using planevec_t = std::vector<pixel_t>;
    using pixvec_t = std::vector<planevec_t>;
    
    enum comparison_t : uint8_t {
        same = 0,                               /// diffvalue < kButteraugliGood
        subtle = 1,                             /// diffvalue > kButteraugliGood && diffvalue < kButteraugliBad
        different = 2,                          /// diffvalue > kButteraugliBad
        error_images_incomprable = 16,          /// ERROR: dimensions of images differ
        error_unexpected_channel_count = 17,    /// ERROR: an operand had the wrong number of channels
        error_augli_not_buttered = 18           /// ERROR: ButteraugliInterface() returned false
    };
    
    const pixel_t EXPONENT = 1.0f / 2.2f;
    
    planevec_t auglize(planevec_t const& in) {
        /// Gamma correction, as per butteraugli source:
        /// https://github.com/google/butteraugli/blob/master/src/butteraugli.h#L40-L44
        const int N = in.size();
        planevec_t out(N);
        auto planerator = out.begin();
        int idx = 0;
        for (auto it = in.begin();
             it != in.end() && idx < N;
             ++it) { planerator = out.emplace(planerator,
                         static_cast<pixel_t>(255.0f * std::pow(*it, EXPONENT)));
                     ++idx; }
        return out;
    }
    
    pixvec_t auglize(pixvec_t const& in) {
        pixvec_t out(in.size());
        std::for_each(in.begin(), in.end(),
                      [&out](planevec_t plane) {
            out.push_back(auglize(plane));
        });
        return out;
    }
    
    comparison_t compare(Image& lhs, Image& rhs) {
        double diffvalue;
        planevec_t diffmap; /// not currently used
        bool augli_buttered = false;
        const int w = lhs.dim(0),
                  h = lhs.dim(1);
        
        if (w != rhs.dim(0) || h != rhs.dim(0)) {
            return error_images_incomprable;
        }
        
        pixvec_t rgb0 = auglize(lhs.allplanes<double>(3)); /// lastplane=3
        pixvec_t rgb1 = auglize(rhs.allplanes<double>(3)); /// lastplane=3
        
        if (rgb0.size() != 3 || rgb1.size() != 3) {
            return error_unexpected_channel_count;
        }
        
        augli_buttered = ButteraugliInterface(w, h, rgb0, rgb1,
                                              diffmap, diffvalue);
        
        if (!augli_buttered) { return error_augli_not_buttered; }
        return diffvalue < kButteraugliGood ? same : 
               diffvalue < kButteraugliBad  ? subtle : different;
        
    }
    
} /* namespace butteraugli */


#endif /// LIBIMREAD_HASHING_HH_