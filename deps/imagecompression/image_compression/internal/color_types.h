// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IMAGE_COMPRESSION_INTERNAL_COLOR_TYPES_H_
#define IMAGE_COMPRESSION_INTERNAL_COLOR_TYPES_H_

#include <algorithm>

#include <cstddef>
#include "image_compression/internal/logging.h"

namespace imagecompression {

    //
    // This file contains some basic types that help deal with colors in image
    // compression and decompression functions.
    //

    //-----------------------------------------------------------------------------

    // Simple templated RGB color type.
    template <typename T>
    struct RgbColor {
        // The default constructor initializes to 0.
        RgbColor()
            : r(0)
            , g(0)
            , b(0) {}
        
        // Constructor that sets all components.
        RgbColor(T red, T green, T blue)
            : r(red)
            , g(green)
            , b(blue) {}
        
        // Accesses a component by index, which can be useful in loops. This is
        // range-checked and emits range errors in debug builds.
        T operator[](int index) const {
            switch (index) {
                case 0: return r;
                case 1: return g;
                case 2: return b;
                default:
                    DCHECK(false) << "Invalid index " << index << " for RGB color";
                    return T(0);
            }
        }
        
        // Unary addition.
        RgbColor& operator+=(RgbColor const& c) {
            r += c.r;
            g += c.g;
            b += c.b;
            return *this;
        }
        
        // Equality operator for testing.
        bool operator==(RgbColor const& c) const { return r == c.r && g == c.g && b == c.b; }
        
        T r;
        T g;
        T b;
    };
    
    // Simple templated RGBA color type.
    template <typename T>
    struct RgbaColor : public RgbColor<T> {
        // The default constructor initializes to 0.
        RgbaColor()
            : a(0) {}
        
        // Constructor that sets all components.
        RgbaColor(T red, T green, T blue, T alpha)
            : RgbColor<T>(red, green, blue)
            , a(alpha) {}
        
        // Constructor taking an RGB color and alpha.
        RgbaColor(RgbColor<T> const& rgb, T alpha)
            : RgbColor<T>(rgb)
            , a(alpha) {}
        
        // Accesses a component by index, which can be useful in loops. This is
        // range-checked and emits range errors in debug builds.
        T operator[](int index) const {
            switch (index) {
                case 0: return this->r;
                case 1: return this->g;
                case 2: return this->b;
                case 3: return a;
                default:
                    DCHECK(false) << "Invalid index " << index << " for RGBA color";
                    return T(0);
            }
        }
        
        // Unary addition.
        RgbaColor& operator+=(RgbaColor const& c) {
            RgbColor<T>::operator+=(c);
            a += c.a;
            return *this;
        }
        
        // Equality operator for testing.
        bool operator==(RgbaColor const& c) const { return RgbColor<T>::operator==(c) && a == c.a; }
        
        T a;
    };
    
    // Defines an RGB pixel stored as 5 bits of red, 6 bits of green, and
    // 5 bits of blue for a total of 16 bits, which fits in a uint16.  It
    // uses endian-specific bit fields to allow casting to be used for
    // very efficient conversions to and from uint16's.  This efficiency
    // is required for performance-critical sections of compression and
    // decompression functions.
    struct Rgb565 {
        // Empty constructor does no initialization.
        Rgb565() {}
        
        // Constructor taking 3 component values assumed to be in range.
        Rgb565(uint8_t red, uint8_t green, uint8_t blue) {
            DCHECK_LT(red, 1 << 5);
            DCHECK_LT(green, 1 << 6);
            DCHECK_LT(blue, 1 << 5);
            // Use assignment instead of initializers because the order of the
            // variables changes with endianness.
            r = red;
            g = green;
            b = blue;
        }
        
        // Equality operator for testing.
        bool operator==(Rgb565 const& c) const { return r == c.r && g == c.g && b == c.b; }
        
#if defined __LITTLE_ENDIAN__
        unsigned int b : 5;
        unsigned int g : 6;
        unsigned int r : 5;
#elif defined __BIG_ENDIAN__
        unsigned int r : 5;
        unsigned int g : 6;
        unsigned int b : 5;
#else
#error Neither __LITTLE_ENDIAN__ nor __BIG_ENDIAN__ is defined
#endif
    };
    
    //-----------------------------------------------------------------------------
    
    //
    // Type-specific instantiations.
    //
    
    // 8-bit-per-component RGB and RGBA colors.
    typedef RgbColor<uint8_t>  Rgb888;
    typedef RgbaColor<uint8_t> Rgba8888;
    
    // Int-component versions of RGB and RGBA colors. They are useful for holding
    // intermediate results of color computations.
    typedef RgbColor<int32_t>  RgbInt;
    typedef RgbaColor<int32_t> RgbaInt;
    
} // namespace imagecompression

#endif // IMAGE_COMPRESSION_INTERNAL_COLOR_TYPES_H_
