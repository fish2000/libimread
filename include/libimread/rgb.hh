/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_RGB_HH_
#define LIBIMREAD_RGB_HH_

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/image.hh>
#include <libimread/metadata.hh>
#include <libimread/ext/valarray.hh>

#define UNCOMPAND(x)    ((x) & 0xFF)
#define R(x)            (UNCOMPAND(x))
#define G(x)            (UNCOMPAND(x >> 8 ))
#define B(x)            (UNCOMPAND(x >> 16))
#define A(x)            (UNCOMPAND(x >> 24))

namespace im {
    
    using MetaImage = ImageWithMetadata;
    using bytearray_t = std::valarray<byte>;
    
    class ByteArray : public Image, public MetaImage {
        
        public:
            
            ByteArray() {}
            
            /// 0: BACKED BY std::valarray<byte>
            
            virtual void* rowp(int r) const override {}
            virtual int nbits() const override {}
            
            virtual int ndims() const override {}
            virtual int dim(int) const override {}
            virtual int stride(int) const override {}
            virtual int min(int) const override {}
            virtual bool is_signed() const override {}
            virtual bool is_floating_point() const override {}
            
            /// 1) MAYBE IMPROVE ON ACCESSOR?!?
            /// 2) 
            
        protected:
            bytearray_t bytearray;
            
    };
    
    class InterleavedImage : public ByteArray {
        
        public:
            
            InterleavedImage()
                :ByteArray()
                {}
            
        private:
            
            struct pixel_t {
                /// pack and unpack variadic template methods
                
                template <typename ...Args>
                int32_t compand(Args... args) const {
                    
                }
                
            };
            
            struct RGB : public pixel_t {
                byte r, g, b;
            };
            
            struct RGBA : public pixel_t {
                byte r, g, b, a;
            };
        
    };
    
    
} /// namespace im


#endif /// LIBIMREAD_RGB_HH_