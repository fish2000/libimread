// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#include <algorithm>
#include <memory>
#include <cmath>
#include <cfenv>

#include <Halide.h>

#include <libimread/IO/ansi.hh>
#include <libimread/image.hh>
#include <libimread/pixels.hh>

#pragma STDC FENV_ACCESS ON

#define XTERM_GRAY_OFFSET       232
#define ADJUST_FACTOR           1.5
// #define SCALE                   256 / 24.0
#define SCALE                   10.6666666667

namespace im {
    
    //using Octet = std::array<uint8_t, 8>;
    using ByteImage = Halide::Image<uint8_t>;
    
    namespace detail {
        
        template <typename T, typename T0 = T> inline
        T distance(T v, T0 v0) {
            return std::fabs(v - v0);
        }
        
        template <int ROUND = FE_DOWNWARD>
        std::uint8_t encode_gray(int value) {
            int current_round = std::fegetround();
            if (ROUND != current_round) { std::fesetround(ROUND); }
            auto out = std::lrint(static_cast<float>(value) / SCALE) + XTERM_GRAY_OFFSET;
            if (ROUND != current_round) { std::fesetround(current_round); }
            return out;
        }
        
        std::uint8_t decode_gray(int value) {
            return static_cast<std::uint8_t>(
                static_cast<float>(value - XTERM_GRAY_OFFSET) * XTERM_GRAY_OFFSET);
        }
        
        std::wchar_t dot(long long bits) {
            return (((bits & 0b111) |
                     (bits & 0b1110000) >> 0b1 |
                     (bits & 0b1000) << 0b11 |
                     (bits & 0b10000000)) + 0x2800);
        }
        
    }
    
    class UnicodeByteArray : public ByteImage, public Image {
        
        public:
            UnicodeByteArray()
                :ByteImage(), Image()
                {}
            
            UnicodeByteArray(int x, int y, int z=8)
                :ByteImage(x/2, y/4, z), Image()
                {}
            
            virtual ~UnicodeByteArray() {}
            
            virtual int nbits() const override {
                /// elem_size is in BYTES, so:
                return sizeof(uint8_t) * 8;
            }
            
            virtual int nbytes() const override {
                return sizeof(uint8_t);
            }
            
            virtual int ndims() const override {
                return ByteImage::dimensions();
            }
            
            virtual int dim(int d) const override {
                return ByteImage::extent(d);
            }
            
            virtual int stride(int s) const override {
                return ByteImage::stride(s);
            }
            
            inline off_t rowp_stride() const {
                return ByteImage::channels() == 1 ? 0 : off_t(ByteImage::stride(1));
            }
            
            virtual void *rowp(int r) override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                uint8_t *host = (uint8_t *)ByteImage::data();
                host += off_t(r * rowp_stride());
                return static_cast<void *>(host);
            }
    };
    
    UnicodeByteArray as_array(Image &input) {
        
    }
    
    
    void ANSIFormat::write(Image &input,
                          byte_sink *output,
                          const options_map &opts) {
    }
    
    void ANSIFormat::write_multi(std::vector<Image> &input,
                                byte_sink *output,
                                const options_map &opts) {
    }
    
}