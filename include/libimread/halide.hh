// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HALIDE_H_
#define LIBIMREAD_HALIDE_H_

#include <cstring>
#include <memory>
#include <vector>
#include <functional>
#include <utility>
#include <stdio.h>
#include <stdint.h>

//#include <Halide.h>

#include <libimread/libimread.hpp>
#include <libimread/private/buffer_t.h>
#include <libimread/errors.hh>
#include <libimread/base.hh>
#include <libimread/tools.hh>

namespace im {
    
    using std::allocator;
    using std::shared_ptr;
    using std::remove_pointer;
    
    static const unsigned int kBUFFERPAD = 40;
    
    template <typename T = uint8_t, typename rT = T>
    inline rT *buffer_new(size_t buffer_size, size_t num=1) {
        if (num < 1 || buffer_size < 1) { return nullptr; }
        /// note value-initialization at the end of next line:
        rT *out = new rT[((sizeof(T) * buffer_size) + kBUFFERPAD) * num]();
        while ((size_t)out & 0x1f) { out++; }
        return out;
    }
    
    template <typename T = uint8_t, typename rT = T>
    inline rT *buffer_alloc(size_t buffer_size, size_t num=1) {
        if (num < 1 || buffer_size < 1) { return nullptr; }
        rT *out = static_cast<rT*>(
            calloc(num, ((sizeof(T) * buffer_size) + kBUFFERPAD)));
        while ((size_t)out & 0x1f) { out++; }
        return out;
    }
    
    inline size_t buffer_size(buffer_t &buf) {
        return buf.extent[0] * buf.extent[1] * buf.extent[2];
    }
    inline size_t buffer_size(const buffer_t &buf) {
        return buf.extent[0] * buf.extent[1] * buf.extent[2];
    }
    
    template <typename T = uint8_t, typename bT = buffer_t>
    struct BufferAllocator : public allocator<T> {
        /// std::allocator_traits boilerplate stuff:
        typedef bT                          value_type;
        typedef bT*                         pointer;
        typedef bT&                         reference;
        typedef void*                       void_pointer;
        typedef const value_type*           const_pointer;
        typedef const value_type&           const_reference;
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
            typedef BufferAllocator<T, U> other;
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
                { 
                    printf("BufferAllocator(): %ix%ix%i\n", w, h, ch);
                }
        
        /// so-called "rebind constructors"
        BufferAllocator(const allocator<T> &a):
            allocator<T>(a),
            _width(1),
            _height(1),
            _channels(1)
                { }
        
        BufferAllocator(const BufferAllocator<T, bT> &ba):
            allocator<T>(ba),
            _width(ba.width()),
            _height(ba.height()),
            _channels(ba.channels())
                { }
        
        ~BufferAllocator() { }
        
        inline unsigned int width() const { return _width; }
        inline void width(unsigned int w) { _width = w; }
        inline unsigned int height() const { return _height; }
        inline void height(unsigned int h) { _height = h; }
        inline unsigned int channels() const { return _channels; }
        inline void channels(unsigned int ch) { _channels = ch; }
        
        inline size_t allocation_size() {
            unsigned int c = _channels > 0 ? _channels : 1;
            return _width * _height * c;
        }
        
        inline pointer allocate(size_type n, const_void_pointer hint=0) {
            printf("BufferAllocator::allocate()\n");
            /// allocation-new for buffer pointer
            pointer alloc_ptr = static_cast<pointer>(calloc(n, sizeof(value_type)));
            return alloc_ptr;
        }
        inline void deallocate(pointer dealloc_ptr, size_type n) {
            printf("BufferAllocator::deallocate()\n");
            if (dealloc_ptr) {
                free(dealloc_ptr);
                dealloc_ptr = nullptr;
            }
        }
        
        template <typename U, typename... Args>
        inline void construct(U* ptr, Args&&... args) {
            printf("BufferAllocator::construct<U, Args...>(): pack size = %i\n", sizeof...(Args));
            construct(ptr, std::forward<Args>(args)...);
        }
        
        inline void construct(pointer ptr, const_reference source) {
            printf("BufferAllocator::construct()\n");
            /// placement-new for entire buffer + room for metadata
            ptr->host_dirty = false;
            ptr->dev_dirty = false;
            ptr->dev = 0;
            
            /// check source buffer pointer(s)
            /// if we have nothing, set up a new buffer,
            /// with the constructed dimensions
            if (!source || !source.host) {
                printf("BufferAllocator::construct(): no source host pointer\n");
                ptr->elem_size = elem_size;
                
                ptr->extent = static_cast<int32_t*>(calloc(4, sizeof(int32_t)));
                ptr->stride = static_cast<int32_t*>(calloc(4, sizeof(int32_t)));
                ptr->min = static_cast<int32_t*>(calloc(4, sizeof(int32_t)));
                ptr->extent[0] = _height;
                ptr->extent[1] = _width;
                ptr->extent[2] = _channels > 0 ? _channels : 1;
                ptr->stride[0] = ptr->extent[2];
                ptr->stride[1] = _height * ptr->extent[2];
                ptr->stride[2] = _width * _height * ptr->extent[2];
                
                size_t size = buffer_size(ptr);
                ptr->host = buffer_alloc<pixel_type>(size);
                return;
            }
            
            /// if we have something, and it's the same size (or larger),
            /// copy everything over that will fit...
            if (source.host) {
                printf("BufferAllocator::construct(): FOUND SOURCE HOST POINTER\n");
                /// ... as long as the element sizes are consistent:
                if (source.elem_size != elem_size) { throw BufferAllocatorError("Element size mismatch"); }
                ptr->elem_size = elem_size;
                
                ptr->extent = static_cast<int32_t*>(calloc(4, sizeof(int32_t)));
                ptr->stride = static_cast<int32_t*>(calloc(4, sizeof(int32_t)));
                ptr->min = static_cast<int32_t*>(calloc(4, sizeof(int32_t)));
                
                size_t ssize = buffer_size(source);
                size_t asize = allocation_size();
                
                if (ssize == asize || ssize < asize) {
                    /// Use their dimensions when sizes are equal,
                    /// or when theirs are smaller
                    ptr->extent[0] = source.extent[0];
                    ptr->extent[1] = source.extent[1];
                    ptr->extent[2] = source.extent[2];
                    ptr->stride[0] = source.stride[0];
                    ptr->stride[1] = source.stride[1];
                    ptr->stride[2] = source.stride[2];
                } else if (ssize > asize) {
                    /// Use our dimensions when source is bigger
                    ptr->extent[0] = _height;
                    ptr->extent[1] = _width;
                    ptr->extent[2] = _channels > 0 ? _channels : 1;
                    ptr->stride[0] = ptr->extent[2];
                    ptr->stride[1] = _height * ptr->extent[2];
                    ptr->stride[2] = _width * _height * ptr->extent[2];
                }
                
                size_t size = buffer_size(ptr);
                ptr->host = buffer_alloc<pixel_type>(size);
                std::memcpy(ptr->host, source.host, size * elem_size);
                return;
            }
        }
        inline void destroy(pointer destroy_ptr) {
            printf("BufferAllocator::destroy()\n");
            if (destroy_ptr->host[0]) {
                free(destroy_ptr->host);
                *destroy_ptr->host = 0;
            }
            if (destroy_ptr->extent[0]) {
                free(destroy_ptr->extent);
                *destroy_ptr->extent = 0;
            }
            if (destroy_ptr->stride[0]) {
                free(destroy_ptr->stride);
                *destroy_ptr->stride = 0;
            }
            if (destroy_ptr->min[0]) {
                free(destroy_ptr->min);
                *destroy_ptr->min = 0;
            }
        }
    };
    
    // template <typename T = uint8_t, typename bT = buffer_t>
    // using shared_buffer = std::shared_ptr<typename remove_pointer<bT>::type, BufferAllocator<T, bT>>;
    
    typedef std::shared_ptr<buffer_t> shared_buffer;
    
    template <typename T = uint8_t, typename bT = buffer_t>
    struct BufferDeleter {
        using allo = BufferAllocator<T, bT>;
        void operator()(bT *buffer_ptr) const {
            allo a(1, 1, 1);
            if (buffer_ptr->host || buffer_ptr->extent || buffer_ptr->stride || buffer_ptr->min) {
                a.destroy(buffer_ptr);
            }
            a.deallocate(buffer_ptr, 1);
        }
    };
    
    template <typename T = uint8_t, typename bT = buffer_t>
    std::shared_ptr<bT> make_shared_buffer(bT buf, unsigned int width=1,
                                                   unsigned int height=1,
                                                   unsigned int channels=1) {
        using allo = const BufferAllocator<T, bT>;
        return std::allocate_shared<bT, allo>(allo(width, height, channels), buf);
    }
    
    template <typename T = uint8_t, typename bT = buffer_t>
    std::shared_ptr<bT> new_shared_buffer(unsigned int width=1,
                                             unsigned int height=1,
                                             unsigned int channels=1) {
        //bT buf = {0};
        //return make_shared_buffer<T, bT>(buf, width, height, channels);
        using allo = BufferAllocator<T, bT>;
        using del = BufferDeleter<T, bT>;
        return std::shared_ptr<bT>(new bT(), del(), allo(width, height, channels));
    }
    
    class HalideBuffer : public Image, public ImageWithMetadata {
        public:
            HalideBuffer(shared_buffer &b, uint8_t nd)
                :buffer(b), ndim(nd)
                { }
            HalideBuffer(const shared_buffer &b, uint8_t nd)
                :buffer(b), ndim(nd)
                { }
            HalideBuffer(shared_buffer &&b, uint8_t nd)
                :buffer(std::move(b)), ndim(nd)
                { }
            HalideBuffer &operator=(const shared_buffer &b) { buffer = b; return *this; }
            HalideBuffer &operator=(shared_buffer &&b) { buffer = std::move(b); return *this; }
            
            /// This may call stuff from Halide.h in the future,
            /// hence the separate implementation
            void finalize();
            uint8_t *release(uint8_t *ptr=nullptr);
            
            ~HalideBuffer() {
                finalize();
            }
            
            virtual int nbits() const override {
                /// elem_size is in BYTES, so:
                return buffer->elem_size * 8;
            }
            
            inline int nbytes() const {
                return buffer->elem_size;
            }
            
            virtual int ndims() const override {
                return ndim;
            }
            
            virtual int dim(int d) const override {
                return static_cast<int>(buffer->extent[d]);
            }
            
            inline int width() const { return buffer->extent[0]; }
            inline int height() const { return buffer->extent[1]; }
            inline int channels() const { return buffer->extent[2]; }
            inline int stride(int dim) const { return buffer->stride[dim]; }
            inline int rowp_stride() const {
                return (channels() == 1 || channels() == 0) ? 0 : stride(1);
            }
            
            void *rowp(int r) override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                uint8_t *host = buffer->host;
                host += (r * rowp_stride());
                return static_cast<void *>(host);
            }
            
            uint8_t &operator()(int x, int y=0, int z=0) {
                // _ASSERT(x < buffer.extents[0], "[imread] X coordinate too large");
                // _ASSERT(y < buffer.extents[1], "[imread] Y coordinate too large");
                // _ASSERT(z < buffer.extents[2], "[imread] Z coordinate too large");
                return buffer->host[nbytes() * (x*stride(0) + y*stride(1) + z*stride(2))];
            }
            
            template <typename T>
            T &at(int x, int y=0, int z=0) {
                return ((T *)buffer->host)[sizeof(T) * (x*stride(0) + y*stride(1) + z*stride(2))];
            }
            
            void set_host_dirty(bool dirty=true) { buffer->host_dirty = dirty; }
            void set_dev_dirty(bool dirty=true) { buffer->dev_dirty = dirty; }
            shared_buffer buf() { return buffer; }
        
        private:
            uint8_t ndim = 0;
            shared_buffer buffer;
    };
    
    template <typename T>
    class HalideBufferFactory : public ImageFactory {
        public:
            typedef T pixel_type;
        
        protected:
            std::unique_ptr<Image> create(int nbits, int d0, int d1, int d2, int d3, int d4) {
                shared_buffer buffer = new_shared_buffer<T, buffer_t>(d0, d1, d2);
                uint8_t ndim = (d2 > 0) ? 3 : 2;
                return std::unique_ptr<Image>(new HalideBuffer(buffer, ndim));
            }
    };
    
    /*
    namespace halide {
        
        template <typename T>
        Halide::Image<T> read(std::string filename) {
            HalideBufferFactory<T> factory;
            options_map opts; /// not currently used when reading
            std::unique_ptr<ImageFormat> format(get_format(split_filename(filename.c_str())));
            
            _ASSERT(format.get(), "[imread] Format is unknown to libimread\n");
            _ASSERT(format->can_read(), "[imread] Format is unreadable by libimread\n");
            
            int fd = ::open(filename.c_str(), O_RDONLY | O_BINARY);
            _ASSERT(!(fd < 0), "[imread] Filesystem/permissions error opening file\n");
            
            std::unique_ptr<byte_source> input(new fd_source_sink(fd));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            HalideBuffer buffer = static_cast<HalideBuffer&>(*output);
            
            fprintf(stderr, "READY!!\n");
            Halide::Image<T> out(*buffer.buf());
            
            _ASSERT(out.data(), "[imread] No data!");
            _ASSERT(out.defined(), "[imread] Output image undefined!");
            
            fprintf(stderr, "Returning image: %ix%ix%i\n",
                out.width(), out.height(), out.channels());
            
            //out.set_host_dirty();
            return out;
        }
        
    }
    */
}

#endif // LIBIMREAD_HALIDE_H_
