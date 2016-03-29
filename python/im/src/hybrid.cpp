
#define NO_IMPORT_ARRAY

#include <cstring>
#include "hybrid.hh"
#include "structcode.hpp"
#include "typecode.hpp"

namespace im {
    
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
        view->buf = rowp(0);
        view->ndim = ndims();
        view->format = ::strdup(detail::structcode(dtype));
        view->shape = new Py_ssize_t[ndims()];
        view->strides = new Py_ssize_t[ndims()];
        view->itemsize = (Py_ssize_t)nbytes();
        view->suboffsets = NULL;
        
        int len = 1;
        for (int idx = 0; idx < view->ndim; idx++) {
            len *= dim_or(idx, 1);
            view->shape[idx] = dim_or(idx, 1);
            view->strides[idx] = stride_or(idx, 1);
        }
        
        view->len = len * nbytes();
        view->readonly = 1; /// true
        view->internal = (void*)"YO DOGG";
        view->obj = NULL;
        
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
    
    PyObject* HalideNumpyImage::metadataPyObject() {
        const std::string& s = MetaImage::get_meta();
        if (s != "") { return PyBytes_FromString(s.c_str()); }
        Py_RETURN_NONE;
    }
    
    /// This returns the same type of data as buffer_t.host
    uint8_t* HalideNumpyImage::data(int s) const {
        return (uint8_t*)HalBase::buffer.host_ptr();
    }
    
    Halide::Type HalideNumpyImage::type() const {
        return HalBase::buffer.type();
    }
    
    buffer_t* HalideNumpyImage::buffer_ptr() const {
        return HalBase::raw_buffer();
    }
    
    int HalideNumpyImage::nbits() const {
        return type().bits();
    }
    
    int HalideNumpyImage::nbytes() const {
        const int bits = nbits();
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
    
    /// extent, stride, min
    NPY_TYPES HalideNumpyImage::dtype()  { return dtype_; }
    int32_t* HalideNumpyImage::dims()    { return HalBase::raw_buffer()->extent; }
    int32_t* HalideNumpyImage::strides() { return HalBase::raw_buffer()->stride; }
    int32_t* HalideNumpyImage::offsets() { return HalBase::raw_buffer()->min; }
    
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