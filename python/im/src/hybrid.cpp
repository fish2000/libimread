
#define NO_IMPORT_ARRAY

#include <cstring>
#include <array>
#include <type_traits>
#include "hybrid.hh"
#include "buffer.hpp"
#include "structcode.hpp"
#include "typecode.hpp"

namespace im {
    
    inline Endian endianness() {
        unsigned long number = 1;
        char* s;
        Endian value = Endian::Unspecified;
        s = (char*)&number;
        if (s[0] == 0) {
            value = Endian::Big;
        } else {
            value = Endian::Little;
        }
        return value;
    }
    
    /// determine system byte order
    const Endian byteorder = im::endianness();
    
    namespace detail {
        
        char const* structcode(NPY_TYPES dtype) {
            switch (dtype) {
                case NPY_BOOL:          return "?";
                case NPY_UINT8:         return "B";
                case NPY_UINT16:        return "H";
                case NPY_UINT32:        return "I";
                case NPY_UINT64:        return "L";
                case NPY_INT8:          return "b";
                case NPY_HALF:          return "e";
                case NPY_INT16:         return "h";
                case NPY_INT32:         return "i";
                case NPY_INT64:         return "l";
                case NPY_FLOAT:         return "f";
                case NPY_DOUBLE:        return "d";
                case NPY_LONGDOUBLE:    return "g";
                default:                return "B";
            }
            return "B";
        }
        
        Halide::Type for_dtype(NPY_TYPES dtype) {
            switch (dtype) {
                case NPY_BOOL:          return Halide::Bool();
                case NPY_UINT8:         return Halide::UInt(8);
                case NPY_UINT16:        return Halide::UInt(16);
                case NPY_UINT32:        return Halide::UInt(32);
                case NPY_INT8:          return Halide::Int(8);
                case NPY_INT16:         return Halide::Int(16);
                case NPY_INT32:         return Halide::Int(32);
                case NPY_FLOAT:         return Halide::Float(32);
                case NPY_DOUBLE:        return Halide::Float(64);
                case NPY_LONGDOUBLE:    return Halide::Float(64);
                default:                return Halide::Handle();
            }
            return Halide::Handle();
        }
        
        NPY_TYPES for_nbits(int nbits, bool signed_type) {
            if (signed_type) {
                switch (nbits) {
                    case 1:             return NPY_BOOL;
                    case 8:             return NPY_INT8;
                    case 16:            return NPY_INT16;
                    case 32:            return NPY_INT32;
                    case 64:            return NPY_INT64;
                }
            } else {
                switch (nbits) {
                    case 1:             return NPY_BOOL;
                    case 8:             return NPY_UINT8;
                    case 16:            return NPY_UINT16;
                    case 32:            return NPY_UINT32;
                    case 64:            return NPY_UINT64;
                }
            }
            return NPY_USERDEF;
        }
        
    }
    
    int PythonBufferImage::populate_buffer(Py_buffer* view,
                                           NPY_TYPES dtype,
                                           int flags) {
        int dimensions          = ndims();
        int bytes               = nbytes();
        view->buf               = rowp(0);
        view->ndim              = dimensions;
        view->format            = ::strdup(detail::structcode(dtype));
        view->shape             = new Py_ssize_t[dimensions];
        view->strides           = new Py_ssize_t[dimensions];
        view->itemsize          = static_cast<Py_ssize_t>(bytes);
        view->suboffsets        = nullptr;
        
        int len = 1;
        for (int idx = 0; idx < dimensions; idx++) {
            int dim_or_one      = dim_or(idx, 1);
            len *= dim_or_one;
            view->shape[idx]    = dim_or_one;
            view->strides[idx]  = stride_or(idx, 1);
        }
        
        view->len               = len * bytes;
        view->readonly          = 1; /// true
        view->internal          = (void*)"YO DOGG";
        view->obj               = nullptr;
        
        /// per the Python API:
        return 0;
    }
    
    void PythonBufferImage::release_buffer(Py_buffer* view) {
        if (std::string((const char*)view->internal) == "YO DOGG") {
            if (view->format)   { std::free(view->format);  view->format  = nullptr; }
            if (view->shape)    { delete[] view->shape;     view->shape   = nullptr; }
            if (view->strides)  { delete[] view->strides;   view->strides = nullptr; }
            view->internal = nullptr;
        }
    }
    
    HalideNumpyImage::HalideNumpyImage()
        :HalBase(), PythonBufferImage(), MetaImage()
        ,dtype_(NPY_UINT8)
        {}
    
    HalideNumpyImage::HalideNumpyImage(NPY_TYPES d, buffer_t const* b, std::string const& name)
        :HalBase(detail::for_dtype(d), b, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HalideNumpyImage::HalideNumpyImage(NPY_TYPES d, int x, int y, int z, int w, std::string const& name)
        :HalBase(detail::for_dtype(d), x, y, z, w, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HalideNumpyImage::HalideNumpyImage(NPY_TYPES d, int x, int y, int z, std::string const& name)
        :HalBase(detail::for_dtype(d), x, y, z, 0, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HalideNumpyImage::HalideNumpyImage(NPY_TYPES d, int x, int y, std::string const& name)
        :HalBase(detail::for_dtype(d), x, y, 0, 0, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HalideNumpyImage::HalideNumpyImage(NPY_TYPES d, int x, std::string const& name)
        :HalBase(detail::for_dtype(d), x, 0, 0, 0, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HalideNumpyImage::HalideNumpyImage(HalideNumpyImage const& other, int zidx, std::string const& name)
        :HalBase(other.type(), other.dim(0), other.dim(1), 1, 0, name)
        ,PythonBufferImage(), MetaImage(name)
        ,dtype_(other.dtype())
        {
            /// rely on Halide's use of planar image strides
            pix::accessor<byte> source = other.access<byte>();
            pix::accessor<byte> target = this->access<byte>();
            const int w = other.dim(0);
            const int h = other.dim(1);
            const int c = zidx;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    pix::convert(source(x, y, c)[0],
                                 target(x, y, 0)[0]);
                }
            }
        }
    
    HalideNumpyImage::HalideNumpyImage(HalideNumpyImage const& basis, HalideNumpyImage const& etc, std::string const& name)
        :HalBase(basis.type(), basis.dim(0), basis.dim(1), basis.dim(2) + etc.dim(2), 0, name)
        ,PythonBufferImage(), MetaImage(name)
        ,dtype_(basis.dtype())
        {
            /// rely on Halide's use of planar image strides
            pix::accessor<byte> source = basis.access<byte>();
            pix::accessor<byte> extend = etc.access<byte>();
            pix::accessor<byte> target = this->access<byte>();
            const int w = basis.dim(0);
            const int h = basis.dim(1);
            const int p = basis.dim(2);
            const int px = etc.dim(2);
            for (int cc = 0; cc < p; ++cc) {
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        pix::convert(source(x, y, cc)[0],
                                     target(x, y, cc)[0]);
                    }
                }
            }
            for (int cc = 0; cc < px; ++cc) {
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        pix::convert(extend(x, y,   cc)[0],
                                     target(x, y, p+cc)[0]);
                    }
                }
            }
        }
    
    HalideNumpyImage::~HalideNumpyImage() {}
    
    /// This returns the same type of data as buffer_t.host
    uint8_t* HalideNumpyImage::data() const {
        return (uint8_t*)HalBase::buffer.host_ptr();
    }
    
    uint8_t* HalideNumpyImage::data(int s) const {
        return (uint8_t*)HalBase::buffer.host_ptr() + std::ptrdiff_t(s);
    }
    
    std::string_view HalideNumpyImage::view() const {
        using value_t = std::add_pointer_t<typename std::string_view::value_type>;
        return std::string_view(static_cast<value_t>(rowp(0)),
                                static_cast<std::size_t>(size()));
    }
    
    Halide::Type HalideNumpyImage::type() const {
        return HalBase::buffer.type();
    }
    
    buffer_t* HalideNumpyImage::buffer_ptr() const {
        return HalBase::raw_buffer();
    }
    
    int HalideNumpyImage::nbits() const {
        return HalBase::buffer.type().bits();
    }
    
    int HalideNumpyImage::nbytes() const {
        const int bits = HalBase::buffer.type().bits();
        return (bits / 8) + bool(bits % 8);
    }
    
    int HalideNumpyImage::ndims() const {
        return HalBase::dimensions();
    }
    
    int HalideNumpyImage::dim(int d) const {
        return HalBase::extent(d);
    }
    
    int HalideNumpyImage::stride(int s) const {
        return HalBase::stride(s);
    }
    
    off_t HalideNumpyImage::rowp_stride() const {
        return HalBase::channels() == 1 ? 0 : off_t(HalBase::stride(1));
    }
    
    void* HalideNumpyImage::rowp(int r) const {
        uint8_t* host = data();
        host += off_t(r * rowp_stride());
        return static_cast<void*>(host);
    }
    
    /// type encoding
    NPY_TYPES   HalideNumpyImage::dtype() const         { return dtype_; }
    char        HalideNumpyImage::dtypechar() const     { return static_cast<char>(typecode::typechar(dtype_)); }
    std::string HalideNumpyImage::dtypename() const     { return typecode::name(dtype_); }
    char const* HalideNumpyImage::structcode() const    { return im::detail::structcode(dtype_); }
    
    std::string HalideNumpyImage::dsignature(Endian e) const {
        char endianness = static_cast<char>(e);
        char typechar = static_cast<char>(typecode::typechar(dtype_));
        int bytes = nbytes();
        int buffer_size = std::snprintf(nullptr, 0, "%c%c%i",
                                        endianness, typechar, bytes) + 1;
        char out_buffer[buffer_size];
        __attribute__((unused))
        int buffer_used = std::snprintf(out_buffer, buffer_size, "%c%c%i",
                                        endianness, typechar, bytes);
        return std::string(out_buffer);
    }
    
    /// extent, stride, min
    int32_t*    HalideNumpyImage::dims()                { return HalBase::raw_buffer()->extent; }
    int32_t*    HalideNumpyImage::strides()             { return HalBase::raw_buffer()->stride; }
    int32_t*    HalideNumpyImage::offsets()             { return HalBase::raw_buffer()->min; }
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    HybridFactory::HybridFactory()
        :nm("Image")
        {}
    
    HybridFactory::HybridFactory(std::string const& n)
        :nm(n)
        {}
    
    HybridFactory::~HybridFactory() {}
    
    std::string& HybridFactory::name() { return nm; }
    void HybridFactory::name(std::string const& n) { nm = n; }
    
    std::unique_ptr<Image> HybridFactory::create(int nbits,
                                                 int xHEIGHT, int xWIDTH, int xDEPTH,
                                                 int d3, int d4) {
        return std::unique_ptr<Image>(
            new HalideNumpyImage(
                detail::for_nbits(nbits), xWIDTH, xHEIGHT, xDEPTH));
    }
    
    std::shared_ptr<Image> HybridFactory::shared(int nbits,
                                                 int xHEIGHT, int xWIDTH, int xDEPTH,
                                                 int d3, int d4) {
        return std::shared_ptr<Image>(
            new HalideNumpyImage(
                detail::for_nbits(nbits), xWIDTH, xHEIGHT, xDEPTH));
    }
    
#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
    
    ArrayImage::ArrayImage()
        :PythonBufferImage(), MetaImage()
        ,array(nullptr), buffer(nullptr)
        {}
    
    static const int FLAGS = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED;
    
    PyTypeObject* newtype() {
        static PyTypeObject* nt = &PyArray_Type;
        return nt;
    }
    
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wmissing-braces"
    ArrayImage::ArrayImage(NPY_TYPES d, buffer_t const* b, std::string const& name)
        :PythonBufferImage(), MetaImage(name)
        ,array(nullptr)
        ,buffer(im::buffer::heapcopy(b))
        ,deallocate(true)
        {
            /// provision dimension/stride arrays for array creation
            std::array<npy_intp, 1> dims1, stride1;
            std::array<npy_intp, 2> dims2, stride2;
            std::array<npy_intp, 3> dims3, stride3;
            std::array<npy_intp, 4> dims4, stride4;
            const int bufferdims   = im::buffer::ndims(*b);
            
            /// Create a new PyArrayObject* with dimensions to match the source buffer,
            /// allocating the underlying storage as needed
            switch (bufferdims) {
                case 1:
                    dims1   = { static_cast<npy_intp>(buffer->extent[0]) };
                    stride1 = { static_cast<npy_intp>(buffer->stride[0]) };
                    array = reinterpret_cast<PyArrayObject*>(
                            PyArray_New(newtype(), bufferdims,
                                  dims1.data(),    d,
                                stride1.data(),    nullptr,
                                buffer->elem_size, FLAGS,
                                nullptr));
                    break;
                case 2:
                    dims2   = { static_cast<npy_intp>(buffer->extent[0]),
                                static_cast<npy_intp>(buffer->extent[1]) };
                    stride2 = { static_cast<npy_intp>(buffer->stride[0]),
                                static_cast<npy_intp>(buffer->stride[1]) };
                    array = reinterpret_cast<PyArrayObject*>(
                            PyArray_New(newtype(), bufferdims,
                                  dims2.data(),    d,
                                stride2.data(),    nullptr,
                                buffer->elem_size, FLAGS,
                                nullptr));
                    break;
                case 3:
                    dims3 =   { static_cast<npy_intp>(buffer->extent[0]),
                                static_cast<npy_intp>(buffer->extent[1]),
                                static_cast<npy_intp>(buffer->extent[2]) };
                    stride3 = { static_cast<npy_intp>(buffer->stride[0]),
                                static_cast<npy_intp>(buffer->stride[1]),
                                static_cast<npy_intp>(buffer->stride[2]) };
                    array = reinterpret_cast<PyArrayObject*>(
                            PyArray_New(newtype(), bufferdims,
                                  dims3.data(),    d,
                                stride3.data(),    nullptr,
                                buffer->elem_size, FLAGS,
                                nullptr));
                    break;
                case 4:
                    dims4 = { static_cast<npy_intp>(buffer->extent[0]),
                              static_cast<npy_intp>(buffer->extent[1]),
                              static_cast<npy_intp>(buffer->extent[2]),
                              static_cast<npy_intp>(buffer->extent[3]) };
                    stride4 = { static_cast<npy_intp>(buffer->stride[0]),
                                static_cast<npy_intp>(buffer->stride[1]),
                                static_cast<npy_intp>(buffer->stride[2]),
                                static_cast<npy_intp>(buffer->stride[3]) };
                    array = reinterpret_cast<PyArrayObject*>(
                            PyArray_New(newtype(), bufferdims,
                                  dims4.data(),    d,
                                stride4.data(),    nullptr,
                                buffer->elem_size, FLAGS,
                                nullptr));
                    break;
            }
            
            /// Copy data to the new array's freshly-allocated storage,
            /// using the source buffer's pointer and calculated length
            std::memcpy(PyArray_DATA(array), (const void*)b->host, im::buffer::length(*b));
            
            /// Point the local buffer copy at the array's now-populated storage
            buffer->host = reinterpret_cast<uint8_t*>(PyArray_BYTES(array));
        }
    
    ArrayImage::ArrayImage(NPY_TYPES d, int x, int y, int z, int w, std::string const& name)
        :PythonBufferImage(), MetaImage(name)
        ,array(nullptr)
        ,buffer(nullptr)
        ,deallocate(true)
        {
            std::array<npy_intp, 4> dimensions{ x, y, z, w };
            std::array<npy_intp, 4> stridings{ 1, x, x*y, x*y*z };
            
            array = reinterpret_cast<PyArrayObject*>(
                    PyArray_New(newtype(),
                        dimensions.size(),
                        dimensions.data(), d,
                        stridings.data(), nullptr,
                        detail::for_dtype(d).bytes(),
                        FLAGS, nullptr));
            
            buffer = new buffer_t{ 0,
                reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                { x, y, z, w },
                {
                    static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 2)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 3))
                },
                { 0, 0, 0, 0 },
                static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                false, false
            };
        }
    
    ArrayImage::ArrayImage(NPY_TYPES d, int x, int y, int z, std::string const& name)
        :PythonBufferImage(), MetaImage(name)
        ,array(nullptr)
        ,buffer(nullptr)
        ,deallocate(true)
        {
            std::array<npy_intp, 3> dimensions{ x, y, z };
            std::array<npy_intp, 3> stridings{ 1, x, x*y };
            
            array = reinterpret_cast<PyArrayObject*>(
                    PyArray_New(newtype(),
                        dimensions.size(),
                        dimensions.data(), d,
                        stridings.data(), nullptr,
                        detail::for_dtype(d).bytes(),
                        FLAGS, nullptr));
            
            buffer = new buffer_t{ 0,
                reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                { x, y, z, 0 },
                {
                    static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 2)),
                    0
                },
                { 0, 0, 0, 0 },
                static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                false, false
            };
        }
    
    ArrayImage::ArrayImage(NPY_TYPES d, int x, int y, std::string const& name)
        :PythonBufferImage(), MetaImage(name)
        ,array(nullptr)
        ,buffer(nullptr)
        ,deallocate(true)
        {
            std::array<npy_intp, 2> dimensions{ x, y };
            std::array<npy_intp, 2> stridings{ 1, x };
            
            array = reinterpret_cast<PyArrayObject*>(
                    PyArray_New(newtype(),
                        dimensions.size(),
                        dimensions.data(), d,
                        stridings.data(), nullptr,
                        detail::for_dtype(d).bytes(),
                        FLAGS, nullptr));
            
            buffer = new buffer_t{ 0,
                reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                { x, y, 0, 0 },
                {
                    static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                    0, 0
                },
                { 0, 0, 0, 0 },
                static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                false, false
            };
        }
    
    ArrayImage::ArrayImage(NPY_TYPES d, int x, std::string const& name)
        :PythonBufferImage(), MetaImage(name)
        ,array(nullptr)
        ,buffer(nullptr)
        ,deallocate(true)
        {
            std::array<npy_intp, 1> dimensions{ x };
            std::array<npy_intp, 1> stridings{ 1 };
            
            array = reinterpret_cast<PyArrayObject*>(
                    PyArray_New(newtype(),
                        dimensions.size(),
                        dimensions.data(), d,
                        stridings.data(), nullptr,
                        detail::for_dtype(d).bytes(),
                        FLAGS, nullptr));
            
            buffer = new buffer_t{ 0,
                reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                { x, 0, 0, 0 },
                {
                    static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                    0, 0, 0
                },
                { 0, 0, 0, 0 },
                static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                false, false
            };
        }
    #pragma clang diagnostic pop
    
    ArrayImage::ArrayImage(ArrayImage const& other)
        :PythonBufferImage(), MetaImage(other.get_meta())
        ,array(reinterpret_cast<PyArrayObject*>(
               PyArray_NewLikeArray(other.array, NPY_KEEPORDER, nullptr, 0)))
        ,buffer(im::buffer::heapcopy(other.buffer))
        ,deallocate(true)
        {
            PyArray_CopyInto(array, other.array);
            
            /// Point the local buffer copy at the array's now-populated storage
            buffer->host = reinterpret_cast<uint8_t*>(PyArray_BYTES(array));
        }
    
    ArrayImage::ArrayImage(ArrayImage&& other) noexcept
        :PythonBufferImage(), MetaImage(std::move(other.get_meta()))
        ,array(reinterpret_cast<PyArrayObject*>(
               PyArray_NewLikeArray(other.array, NPY_KEEPORDER, nullptr, 0)))
        ,buffer(std::move(other.buffer))
        ,deallocate(true)
        {
            PyArray_MoveInto(array, other.array);
            other.deallocate = false;
            
            /// Point the local buffer copy at the array's now-populated storage
            buffer->host = reinterpret_cast<uint8_t*>(PyArray_BYTES(array));
        }
    
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wmissing-braces"
    ArrayImage::ArrayImage(ArrayImage const& other, int zidx, std::string const& name)
        :PythonBufferImage(), MetaImage(name)
        ,array(nullptr)
        ,buffer(nullptr)
        ,deallocate(true)
        {
            const int x = other.dim(0);
            const int y = other.dim(1);
            const int z = 1;
            const NPY_TYPES d = other.dtype();
            std::array<npy_intp, 3> dimensions{ x, y, z };
            std::array<npy_intp, 3> stridings{ 1, x, x*y };
            
            array = reinterpret_cast<PyArrayObject*>(
                    PyArray_New(newtype(),
                        dimensions.size(),
                        dimensions.data(), d,
                        stridings.data(), nullptr,
                        detail::for_dtype(d).bytes(),
                        FLAGS, nullptr));
            
            buffer = new buffer_t{ 0,
                reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                { x, y, z, 0 },
                {
                    static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 2)),
                    0
                },
                { 0, 0, 0, 0 },
                static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                false, false
            };
            
            /// rely on Halide's use of planar image strides
            pix::accessor<byte> source = other.access<byte>();
            pix::accessor<byte> target = this->access<byte>();
            const int c = zidx;
            for (int yy = 0; yy < y; ++yy) {
                for (int xx = 0; xx < x; ++xx) {
                    pix::convert(source(xx, yy, c)[0],
                                 target(xx, yy, 0)[0]);
                }
            }
        }
    
    ArrayImage::ArrayImage(ArrayImage const& basis, ArrayImage const& etc, std::string const& name)
        :PythonBufferImage(), MetaImage(name)
        ,array(nullptr)
        ,buffer(nullptr)
        ,deallocate(true)
        {
            const int x = basis.dim(0);
            const int y = basis.dim(1);
            const int z = basis.dim(2) + etc.dim(2);
            const NPY_TYPES d = basis.dtype();
            std::array<npy_intp, 3> dimensions{ x, y, z };
            std::array<npy_intp, 3> stridings{ 1, x, x*y };
            
            array = reinterpret_cast<PyArrayObject*>(
                    PyArray_New(newtype(),
                        dimensions.size(),
                        dimensions.data(), d,
                        stridings.data(), nullptr,
                        detail::for_dtype(d).bytes(),
                        FLAGS, nullptr));
            
            buffer = new buffer_t{ 0,
                reinterpret_cast<uint8_t*>(PyArray_DATA(array)),
                { x, y, z, 0 },
                {
                    static_cast<int32_t>(PyArray_STRIDE(array, 0)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 1)),
                    static_cast<int32_t>(PyArray_STRIDE(array, 2)),
                    0
                },
                { 0, 0, 0, 0 },
                static_cast<int32_t>(PyArray_ITEMSIZE(array)),
                false, false
            };
            
            /// rely on Halide's use of planar image strides
            pix::accessor<byte> source = basis.access<byte>();
            pix::accessor<byte> extend = etc.access<byte>();
            pix::accessor<byte> target = this->access<byte>();
            const int w = basis.dim(0);
            const int h = basis.dim(1);
            const int p = basis.dim(2);
            const int px = etc.dim(2);
            for (int cc = 0; cc < p; ++cc) {
                for (int yy = 0; yy < h; ++yy) {
                    for (int xx = 0; xx < w; ++xx) {
                        pix::convert(source(xx, yy, cc)[0],
                                     target(xx, yy, cc)[0]);
                    }
                }
            }
            for (int cc = 0; cc < px; ++cc) {
                for (int yy = 0; yy < h; ++yy) {
                    for (int xx = 0; xx < w; ++xx) {
                        pix::convert(extend(xx, yy,   cc)[0],
                                     target(xx, yy, p+cc)[0]);
                    }
                }
            }
        }
    #pragma clang diagnostic pop
    
    ArrayImage::~ArrayImage() {
        if (deallocate) { delete buffer; }
        Py_XDECREF(array);
    }
    
    /// This returns the same type of data as buffer_t.host
    uint8_t* ArrayImage::data() const {
        return (uint8_t*)PyArray_GETPTR1(array, 0);
    }
    
    uint8_t* ArrayImage::data(int s) const {
        return (uint8_t*)PyArray_GETPTR1(array, s);
    }
    
    std::string_view ArrayImage::view() const {
        using value_t = std::add_pointer_t<typename std::string_view::value_type>;
        return std::string_view(static_cast<value_t>(rowp(0)),
                                static_cast<std::size_t>(size()));
    }
    
    Halide::Type ArrayImage::type() const {
        return detail::for_dtype(
            static_cast<NPY_TYPES>(PyArray_TYPE(array)));
    }
    
    buffer_t* ArrayImage::buffer_ptr() const {
        return buffer;
    }
    
    int ArrayImage::nbits() const {
        return type().bits();
    }
    
    int ArrayImage::nbytes() const {
        const int bits = type().bits();
        return (bits / 8) + bool(bits % 8);
    }
    
    int ArrayImage::ndims() const {
        return PyArray_NDIM(array);
    }
    
    int ArrayImage::dim(int d) const {
        return PyArray_DIM(array, d);
    }
    
    int ArrayImage::stride(int s) const {
        return PyArray_STRIDE(array, s);
    }
    
    void* ArrayImage::rowp(int r) const {
        return PyArray_GETPTR1(array, r);
    }
    
    /// type encoding
    NPY_TYPES   ArrayImage::dtype() const         { return static_cast<NPY_TYPES>(PyArray_TYPE(array)); }
    char        ArrayImage::dtypechar() const     { return static_cast<char>(typecode::typechar(PyArray_TYPE(array))); }
    std::string ArrayImage::dtypename() const     { return typecode::name(PyArray_TYPE(array)); }
    char const* ArrayImage::structcode() const    { return im::detail::structcode(
                                                           static_cast<NPY_TYPES>(PyArray_TYPE(array))); }
    
    std::string ArrayImage::dsignature(Endian e) const {
        char endianness = static_cast<char>(e);
        char typechar = static_cast<char>(typecode::typechar(PyArray_TYPE(array)));
        int bytes = nbytes();
        int buffer_size = std::snprintf(nullptr, 0, "%c%c%i",
                                        endianness, typechar, bytes) + 1;
        char out_buffer[buffer_size];
        __attribute__((unused))
        int buffer_used = std::snprintf(out_buffer, buffer_size, "%c%c%i",
                                        endianness, typechar, bytes);
        return std::string(out_buffer);
    }
    
    /// extent, stride, min
    int32_t*    ArrayImage::dims()                { return buffer->extent; }
    int32_t*    ArrayImage::strides()             { return buffer->stride; }
    int32_t*    ArrayImage::offsets()             { return buffer->min; }
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    ArrayFactory::ArrayFactory()
        :nm("Array")
        {}
    
    ArrayFactory::ArrayFactory(std::string const& n)
        :nm(n)
        {}
    
    ArrayFactory::~ArrayFactory() {}
    
    std::string& ArrayFactory::name() { return nm; }
    void ArrayFactory::name(std::string const& n) { nm = n; }
    
    std::unique_ptr<Image> ArrayFactory::create(int nbits,
                                                int xHEIGHT, int xWIDTH, int xDEPTH,
                                                int d3, int d4) {
        return std::unique_ptr<Image>(
            new ArrayImage(
                detail::for_nbits(nbits), xWIDTH, xHEIGHT, xDEPTH));
    }
    
    std::shared_ptr<Image> ArrayFactory::shared(int nbits,
                                                int xHEIGHT, int xWIDTH, int xDEPTH,
                                                int d3, int d4) {
        return std::shared_ptr<Image>(
            new ArrayImage(
                detail::for_nbits(nbits), xWIDTH, xHEIGHT, xDEPTH));
    }
    
#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
    

}