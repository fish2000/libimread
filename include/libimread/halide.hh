// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HALIDE_H_
#define LIBIMREAD_HALIDE_H_

#include <memory>
#include <vector>
#include <functional>
#include <utility>
#include <stdio.h>
#include <stdint.h>

#include <libimread/libimread.hpp>
#include <libimread/private/buffer_t.h>
#include <libimread/errors.hh>
#include <libimread/base.hh>
#include <libimread/tools.hh>

namespace im {
    
    using std::allocator;
    
    static const unsigned int kBUFFERPAD = 40;
        
    template <typename T = uint8_t, typename rT = T>
    rT *buffer_alloc(unsigned int buffer_size, unsigned int num=1) {
        if (num < 1 || buffer_size < 1) { return nullptr; }
        /// note value-initialization at the end of next line:
        return new rT[((sizeof(T) * buffer_size) * num) + kBUFFERPAD]();
    }
    
    template <typename T>
    struct BufferAllocator : public allocator<T> {
        /// std::allocator_traits boilerplate stuff:
        typedef buffer_t                    value_type;
        typedef buffer_t*                   pointer;
        typedef buffer_t&                   reference;
        typedef void*                       void_pointer;
        typedef const buffer_t*             const_pointer;
        typedef const buffer_t&             const_reference;
        typedef const void*                 const_void_pointer;
        typedef size_t                      size_type;
        typedef ptrdiff_t                   difference_type;
        
        /// OUR SHIT:
        typedef T                           pixel_type;
        static constexpr int elem_size      = sizeof(T);
        
        /// OUR MUTABLE SHIT:
        unsigned int _width                 = 0;
        unsigned int _height                = 0;
        unsigned int _channels              = 0;
        // unsigned int _stride_x       = 0;
        // unsigned int _stride_y       = 0;
        // unsigned int _stride_z       = 0;
        
        template <typename U>
        struct rebind {
            /// this may be horribly wrong
            typedef BufferAllocator<U> other;
        };
        
        BufferAllocator():
            allocator<T>(),
            _width(1),
            _height(1),
            _channels(1)
                { }
        
        BufferAllocator(unsigned int w=1,
                        unsigned int h=1,
                        unsigned int ch=1):
            allocator<T>(),
            _width(w),
            _height(h),
            _channels(ch)
                { }
        
        /// so-called "rebind constructors"
        BufferAllocator(const allocator<T> &a):
            allocator<T>(),
            _width(1),
            _height(1),
            _channels(1)
                { }
        
        BufferAllocator(const BufferAllocator<T> &a):
            allocator<T>(a),
            _width(a.width()),
            _height(a.height()),
            _channels(a.channels())
                { }
        
        ~BufferAllocator() { }
        
        inline unsigned int width() { return _width; }
        inline void width(unsigned int w) { _width = w; }
        inline unsigned int height() { return _height; }
        inline void height(unsigned int h) { _height = h; }
        inline unsigned int channels() { return _channels; }
        inline void channels(unsigned int ch) { _channels = ch; }
        
        inline pointer allocate(size_type n, const_void_pointer hint=0) {
            /// allocation-new for buffer pointer
            pointer alloc_ptr = buffer_alloc<pixel_type>(n);
            return alloc_ptr;
        }
        inline void deallocate(pointer dealloc_ptr, size_type n) {
            delete[] dealloc_ptr;
        }
        
        inline void construct(pointer p, const_reference source) {
            /// placement-new for entire buffer + room for metadata
            uint8_t ndim = (_channels > 0) ? 3 : 2;
            buffer_t buffer = {0};
            buffer.elem_size = elem_size;
            
            buffer.extent[0] = _height;
            buffer.extent[1] = _width;
            buffer.extent[2] = _channels > 0 ? _channels : 1;
            
            buffer.stride[0] = buffer.extent[2];
            buffer.stride[1] = _height * buffer.extent[2];
            buffer.stride[2] = _width * _height * buffer.extent[2];
            
            buffer.min[0] = 0;
            buffer.min[1] = 0;
            buffer.min[2] = 0;
            
            size_t size = buffer.extent[0] * buffer.extent[1] * buffer.extent[2];
            //uint8_t *alloc_ptr = new uint8_t[sizeof(T) * size + 40];
            //buffer.host = alloc_ptr;
            //buffer.host = new uint8_t[sizeof(T) * size + 40];
            buffer.dev = 0;
            buffer.host_dirty = false;
            buffer.dev_dirty = false;
            
        }
        inline void destroy(pointer p) {}
        
    };
    
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
            uint8_t *release(uint8_t *ptr=nullptr);
            
            ~HalideBuffer() {
                finalize();
            }
            
            virtual int nbits() const override {
                /// elem_size is in BYTES, so:
                return buffer.elem_size * 8;
            }
            
            inline int nbytes() const {
                return buffer.elem_size;
            }
            
            virtual int ndims() const override {
                return ndim;
            }
            
            virtual int dim(int d) const override {
                return static_cast<int>(buffer.extent[d]);
            }
            
            inline int width() const { return buffer.extent[0]; }
            inline int height() const { return buffer.extent[1]; }
            inline int channels() const { return buffer.extent[2]; }
            inline int stride(int dim) const { return buffer.stride[dim]; }
            inline int rowp_stride() const {
                return (channels() == 1 || channels() == 0) ? 0 : stride(1);
            }
            
            void *rowp(int r) override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                uint8_t *host = buffer.host;
                host += (r * rowp_stride());
                return static_cast<void *>(host);
            }
            
            uint8_t &operator()(int x, int y=0, int z=0) {
                // _ASSERT(x < buffer.extents[0], "[imread] X coordinate too large");
                // _ASSERT(y < buffer.extents[1], "[imread] Y coordinate too large");
                // _ASSERT(z < buffer.extents[2], "[imread] Z coordinate too large");
                return buffer.host[nbytes() * (x*stride(0) + y*stride(1) + z*stride(2))];
            }
            
            template <typename T>
            T &at(int x, int y=0, int z=0) {
                return ((T *)buffer.host)[sizeof(T) * (x*stride(0) + y*stride(1) + z*stride(2))];
            }
            
            void set_host_dirty(bool dirty=true) { buffer.host_dirty = dirty; }
            void set_dev_dirty(bool dirty=true) { buffer.dev_dirty = dirty; }
            operator buffer_t *() const { return const_cast<buffer_t *>(&buffer); }
            buffer_t *buf() const { return const_cast<buffer_t *>(&buffer); }
            uint8_t *data() { return allocation; }
        
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
            std::unique_ptr<Image> create(int nbits, int d0, int d1, int d2, int d3, int d4) {
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
                //buffer.host = new uint8_t[sizeof(T) * size + 40];
                buffer.dev = 0;
                buffer.host_dirty = false;
                buffer.dev_dirty = false;
                
                while ((size_t)buffer.host & 0x1f) { buffer.host++; }
                return std::unique_ptr<Image>(
                    new HalideBuffer(std::move(buffer), ndim, alloc_ptr));
            }
    };

}

#endif // LIBIMREAD_HALIDE_H_
