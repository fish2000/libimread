// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HALIDE_H_
#define LIBIMREAD_HALIDE_H_

#include <memory>
#include <utility>
#include <stdio.h>
#include <stdint.h>
#include "private/buffer_t.h"
#include "base.h"

namespace im {

    class HalideBuffer : public Image, public ImageWithMetadata {
        public:
            HalideBuffer(buffer_t &b, uint8_t nd, uint8_t *a=nullptr)
                :buffer(b), ndim(nd), allocation(a)
                { }
            HalideBuffer(const buffer_t &b, uint8_t nd, uint8_t *a=nullptr)
                :buffer(b), ndim(nd), allocation(a)
                { }
            HalideBuffer(buffer_t &&b, uint8_t nd, uint8_t *a=nullptr)
                :buffer(std::move(b)), ndim(nd), allocation(a)
                { }
            HalideBuffer &operator=(const buffer_t &b) { buffer = b; return *this; }
            HalideBuffer &operator=(buffer_t &&b) { buffer = std::move(b); return *this; }
            
            /// This may call stuff from Halide.h in the future,
            /// hence the separate implementation
            void finalize();
            
            ~HalideBuffer() {
                finalize();
            }
            
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
                uint8_t *host = buffer.host;
                host += (r * buffer.stride[0] * buffer.elem_size);
                return static_cast<void *>(host);
            }
            
            void set_host_dirty(bool dirty=true) { buffer.host_dirty = dirty; }
            void set_dev_dirty(bool dirty=true) { buffer.dev_dirty = dirty; }
            operator buffer_t *() const { return const_cast<buffer_t *>(&buffer); }
            buffer_t *buf() const { return const_cast<buffer_t *>(&buffer); }
        
        private:
            uint8_t ndim = 0;
            uint8_t *allocation = nullptr;
            buffer_t buffer = {0};
    };
    
    template <typename T>
    class HalideBufferFactory : public ImageFactory {
        public:
            typedef T type;
        
        protected:
            std::auto_ptr<Image> create(int nbits, int d0, int d1, int d2, int d3, int d4) {
                uint8_t ndim = (d2 > 0) ? 3 : 2;
                buffer_t buffer = {0};
                buffer.elem_size = sizeof(T);
            
                buffer.extent[0] = d0;
                buffer.extent[1] = d1;
                buffer.extent[2] = d2 > 0 ? d2 : 1;
            
                buffer.stride[0] = buffer.extent[2];
                buffer.stride[1] = d0 * buffer.extent[2];
                buffer.stride[2] = d1 * d0 * buffer.extent[2];
                
                buffer.min[0] = 0;
                buffer.min[1] = 0;
                buffer.min[2] = 0;
                
                size_t size = buffer.extent[0] * buffer.extent[1] * buffer.extent[2];
                uint8_t *alloc_ptr = new uint8_t[sizeof(T) * size + 40];
                buffer.host = alloc_ptr;
                buffer.dev = 0;
                buffer.host_dirty = false;
                buffer.dev_dirty = false;
                
                while ((size_t)buffer.host & 0x1f) { buffer.host++; }
                return std::auto_ptr<Image>(
                    new HalideBuffer(
                        std::move(buffer),
                        ndim, alloc_ptr));
            }
    };

}

#endif // LIBIMREAD_HALIDE_H_
