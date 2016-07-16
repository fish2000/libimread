
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <libimread/hashing.hh>
#include <libimread/ext/butteraugli.hh>

namespace blockhash {
    
    namespace orig {
        
        int cmpint(const void* pa, const void* pb) noexcept {
            int a = *(const int *) pa;
            int b = *(const int *) pb;
            return (a < b) ? -1 : (a > b);
        }
        
        int cmpfloat(const void* pa, const void* pb) noexcept {
            float a = *(const float *) pa;
            float b = *(const float *) pb;
            return (a < b) ? -1 : (a > b);
        }
        
        float median(int* data, int n) {
            int *sorted;
            float result;
            
            sorted = (int *)malloc(n * sizeof(int));
            std::memcpy(sorted, data, n * sizeof(int));
            std::qsort(sorted, n, sizeof(int), cmpint);
            
            if (n % 2 == 0) {
                result = static_cast<float>(sorted[n / 2] + sorted[n / 2 + 1]) / 2;
            } else {
                result = static_cast<float>(sorted[n / 2]);
            }
            free(sorted);
            return result;
        }
        
        float medianf(float* data, int n) {
            float *sorted;
            float result;
            
            sorted = (float *)malloc(n * sizeof(float));
            std::memcpy(sorted, data, n * sizeof(float));
            std::qsort(sorted, n, sizeof(float), cmpfloat);
            
            if (n % 2 == 0) {
                result = (sorted[n / 2] + sorted[n / 2 + 1]) / 2;
            } else {
                result = sorted[n / 2];
            }
            free(sorted);
            return result;
        }
        
        /** Convert array of bits to hexadecimal string representation.
         * Hash length should be a multiple of 4.
         *
         * Returns: null-terminated hexadecimal string hash.
         */
        char* bits_to_hexhash(int* bits, int nbits) {
            int    i, j, b;
            int    len;
            int    tmp;
            char  *hex;
            char  *stmp;
            
            len = nbits / 4;
            
            hex = (char *)malloc(len + 1);
            stmp = (char *)malloc(2);
            hex[len] = '\0';
            
            for (i = 0; i < len; i++) {
                tmp = 0;
                for (j = 0; j < 4; j++) {
                    b = i * 4 + j;
                    tmp = tmp | (bits[b] << 3 >> j);
                }
                std::sprintf(stmp, "%1x", tmp);
                hex[i] = stmp[0];
            }
            free(stmp);
            return hex;
        }
        
        /** Calculate perceptual hash for an RGBA image using quick method.
        *
        * Quick method uses rounded block sizes and is less accurate in case image
        * width and height are not divisible by the number of bits.
        *
        * Parameters:
        *
        * bits - number of blocks to divide the image by horizontally and vertically.
        * data - RGBA image data.
        * width - image width.
        * height - image height.
        * hash - the resulting hash will be allocated and stored in the given array as bits.
        */
        void blockhash_quick(int bits, unsigned char* data,
                             int width, int height, int** hash) {
            int    i, x, y, ix, iy;
            int    ii, alpha, value;
            int    block_width;
            int    block_height;
            int   *blocks;
            float  m[4];
            
            block_width = width / bits;
            block_height = height / bits;
            
            blocks = (int *)calloc(bits * bits, sizeof(int));
            for (y = 0; y < bits; y++) {
                for (x = 0; x < bits; x++) {
                    value = 0;
                    
                    for (iy = 0; iy < block_height; iy++) {
                        for (ix = 0; ix < block_width; ix++) {
                            ii = ((y * block_height + iy) * width + (x * block_width + ix)) * 4;
                            
                            alpha = data[ii+3];
                            if (alpha == 0) {
                                value += 765;
                            } else {
                                value += data[ii] + data[ii+1] + data[ii+2];
                            }
                        }
                    }
                    
                    blocks[y * bits + x] = value;
                }
            }
            
            for (i = 0; i < 4; i++) {
               m[i] = median(&blocks[i*bits*bits/4], bits*bits/4);
            }
            
            for (i = 0; i < bits * bits; i++) {
                if (  ((blocks[i] < m[0]) && (i < bits*bits/4))
                    ||((blocks[i] < m[1]) && (i >= bits*bits/4) && (i < bits*bits/2))
                    ||((blocks[i] < m[2]) && (i >= bits*bits/2) && (i < bits*bits/4+bits*bits/2))
                    ||((blocks[i] < m[3]) && (i >= bits*bits/2+bits*bits/4))
                    ) {
                  blocks[i] = 0;
                } else {
                  blocks[i] = 1;
                }
            }
            
            *hash = blocks;
        }
        
        /** Calculate perceptual hash for an RGBA image using precise method.
        *
        * Precise method puts weighted pixel values to blocks according to pixel
        * area falling within a given block and provides more accurate results
        * in case width and height are not divisible by the number of bits.
        *
        * Parameters:
        *
        * bits - number of blocks to divide the image by horizontally and vertically.
        * data - RGBA image data.
        * width - image width.
        * height - image height.
        * hash - the resulting hash will be allocated and stored in the given array as bits.
        */
        void blockhash(int bits, unsigned char* data,
                       int width, int height, int** hash) {
            float   block_width;
            float   block_height;
            float   y_frac, y_int;
            float   x_frac, x_int;
            float   x_mod, y_mod;
            float   weight_top, weight_bottom, weight_left, weight_right;
            int     block_top, block_bottom, block_left, block_right;
            int     i, x, y, ii, alpha;
            float   value;
            float  *blocks;
            int    *result;
            float   m[4];
            
            if (width % bits == 0 && height % bits == 0) {
                return blockhash_quick(bits, data, width, height, hash);
            }
            
            block_width = (float) width / (float) bits;
            block_height = (float) height / (float) bits;
            
            blocks = (float *)calloc(bits * bits, sizeof(float));
            result = (int *)malloc(bits * bits * sizeof(int));
            
            for (y = 0; y < height; y++) {
                y_mod = std::fmod(y + 1, block_height);
                y_frac = std::modf(y_mod, &y_int);
                
                weight_top = (1 - y_frac);
                weight_bottom = y_frac;
                
                // y_int will be 0 on bottom/right borders and on block boundaries
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
                    
                    // x_int will be 0 on bottom/right borders and on block boundaries
                    if (x_int > 0 || (x + 1) == width) {
                        block_left = block_right = (int)std::floor((float) x / block_width);
                    } else {
                        block_left = (int)std::floor((float) x / block_width);
                        block_right = (int)std::ceil((float) x / block_width);
                    }
                    
                    ii = (y * width + x) * 4;
                    
                    alpha = data[ii + 3];
                    if (alpha == 0) {
                        value = 765;
                    } else {
                        value = data[ii] + data[ii + 1] + data[ii + 2];
                    }
                    
                    // add weighted pixel value to relevant blocks
                    blocks[block_top * bits + block_left] += value * weight_top * weight_left;
                    blocks[block_top * bits + block_right] += value * weight_top * weight_right;
                    blocks[block_bottom * bits + block_left] += value * weight_bottom * weight_left;
                    blocks[block_bottom * bits + block_right] += value * weight_bottom * weight_right;
                }
            }
            
            for (i = 0; i < 4; i++) {
                m[i] = medianf(&blocks[i*bits*bits/4], bits*bits/4);
            }
            
            for (i = 0; i < bits * bits; i++) {
                if (  ((blocks[i] < m[0]) && (i < bits*bits/4))
                   || ((blocks[i] < m[1]) && (i >= bits*bits/4) && (i < bits*bits/2))
                   || ((blocks[i] < m[2]) && (i >= bits*bits/2) && (i < bits*bits/4+bits*bits/2))
                   || ((blocks[i] < m[3]) && (i >= bits*bits/2+bits*bits/4))
                   )
                {
                  result[i] = 0;
                } else {
                  result[i] = 1;
                }
            }
            
            *hash = result;
            free(blocks);
        }
    
    }; /* namespace orig */
    
}; /* namespace blockhash */

namespace butteraugli {
    
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
                      [&out](planevec_t const& plane) {
            out.push_back(std::move(auglize(plane)));
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
        
        pixvec_t rgb0 = std::move(auglize(lhs.allplanes<double>(3))); /// lastplane=3
        pixvec_t rgb1 = std::move(auglize(rhs.allplanes<double>(3))); /// lastplane=3
        
        if (rgb0.size() != 3 || rgb1.size() != 3) {
            return error_unexpected_channel_count;
        }
        
        augli_buttered = ButteraugliInterface(w, h, rgb0, rgb1,
                                              diffmap, diffvalue);
        
        if (!augli_buttered) { return error_augli_not_buttered; }
        return diffvalue < kButteraugliGood ? same : 
               diffvalue < kButteraugliBad  ? subtle : different;
        
    }
    
    comparison_t compare(Image* lhs, Image* rhs) {
        double diffvalue;
        planevec_t diffmap; /// not currently used
        bool augli_buttered = false;
        const int w = lhs->dim(0),
                  h = lhs->dim(1);
        
        if (w != rhs->dim(0) || h != rhs->dim(0)) {
            return error_images_incomprable;
        }
        
        pixvec_t rgb0 = std::move(auglize(lhs->allplanes<double>(3))); /// lastplane=3
        pixvec_t rgb1 = std::move(auglize(rhs->allplanes<double>(3))); /// lastplane=3
        
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