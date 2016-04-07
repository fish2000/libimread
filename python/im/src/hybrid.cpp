
#define NO_IMPORT_ARRAY

#include <cstring>
#include <type_traits>
#include "hybrid.hh"
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
        view->suboffsets        = NULL;
        
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
        view->obj               = NULL;
        
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
    
    HalideNumpyImage::~HalideNumpyImage() {}
    
    // PyObject* HalideNumpyImage::metadataPyObject() {
    //     std::string const& s = MetaImage::get_meta();
    //     if (s != "") { return PyBytes_FromStringAndSize(s.c_str(),
    //                                                     s.size()); }
    //     Py_RETURN_NONE;
    // }
    
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
        :nm("")
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
    
    

}