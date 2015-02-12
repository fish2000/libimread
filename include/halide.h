// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HALIDE_H_
#define LIBIMREAD_HALIDE_H_

#include <memory>
#include <utility>
#include "private/buffer_t.h"
#include "base.h"

namespace im {

    class HalideBuffer : public Image, public ImageWithMetadata {
        public:
            HalideBuffer(buffer_t &b, uint8_t nd)
                :buffer(b), ndim(nd)
                { }
            HalideBuffer(const buffer_t &b, uint8_t nd)
                :buffer(b), ndim(nd)
                { }
            HalideBuffer(buffer_t &&b, uint8_t nd)
                :buffer(std::move(b)), ndim(nd)
                { }
            HalideBuffer &operator=(const buffer_t &b) { buffer = b; return *this; }
            HalideBuffer &operator=(buffer_t &&b) { buffer = std::move(b); return *this; }
            
            ~HalideBuffer() {
                if (buffer.host) {
                    delete buffer.host;
                }
            }
            
            operator buffer_t *() const { return const_cast<buffer_t *>(&buffer); }
            
            virtual int nbits() const override {
                /// elem_size is in BYTES, so:
                return buffer.elem_size * 8;
            }
            
            virtual int ndims() const override {
                return ndim;
            }
            
            virtual int dim(int d) const override {
                return static_cast<int>(buffer.extent[d]);
            }
            
            void *rowp(int r) override {
                /// WARNING: FREAKY POINTER ARITHMETIC FOLLOWS
                uint8_t *host = reinterpret_cast<uint8_t *>(buffer.host);
                host += (r * buffer.stride[1]);
                return reinterpret_cast<void *>(host);
            }
            
            void finalize();
            void set_host_dirty(bool dirty=true) { buffer.host_dirty = dirty; }
            void set_dev_dirty(bool dirty=true) { buffer.dev_dirty = dirty; }
        
        private:
            uint8_t ndim = 0;
            buffer_t buffer = {0};
    };
    
    template <typename T>
    class HalideBufferFactory : public ImageFactory {
        typedef T type;
        
        protected:
            std::auto_ptr<Image> create(int nbits, int d0, int d1, int d2, int d3, int d4) {
                uint8_t ndim = (d2 > 0) ? 3 : 2;
                buffer_t buffer = {0};
                //buffer.elem_size = int32_t(nbits / 8);
                buffer.elem_size = sizeof(T);
                
                buffer.extent[0] = d1;
                buffer.extent[1] = d0;
                buffer.extent[2] = d2 > 0 ? d2 : 1;
                //buffer.extent[3] = d3 > 0 ? d3 : 0;
                //buffer.extent[3] = 0;
                
                buffer.stride[0] = buffer.extent[2];
                buffer.stride[1] = d0 * buffer.extent[2];
                buffer.stride[2] = 1;
                
                buffer.min[0] = 0;
                buffer.min[1] = 0;
                buffer.min[2] = 0;
                
                size_t size = buffer.extent[0] * buffer.extent[1] * buffer.extent[2];
                uint8_t *ptr = new uint8_t[sizeof(T) * size + 40];
                buffer.dev = 0;
                buffer.host = ptr;
                buffer.host_dirty = false;
                buffer.dev_dirty = false;
                
                while ((size_t)buffer.host & 0x1f) { buffer.host++; }
                return std::auto_ptr<Image>(new HalideBuffer(std::move(buffer), ndim));
            }
    };

}

#endif // LIBIMREAD_HALIDE_H_
