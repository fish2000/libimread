
#define NO_IMPORT_ARRAY

#include "numpy.hh"
#include "structcode.hpp"
#include "typecode.hpp"

namespace im {
    
    namespace detail {
        
        Halide::Type for_dtype(NPY_TYPES dtype) {
            switch (dtype) {
                case NPY_BOOL: return Halide::Bool();
                case NPY_UINT8: return Halide::UInt(8);
                case NPY_UINT16: return Halide::UInt(16);
                case NPY_UINT32: return Halide::UInt(32);
                case NPY_INT8: return Halide::Int(8);
                case NPY_INT16: return Halide::Int(16);
                case NPY_INT32: return Halide::Int(32);
                case NPY_FLOAT: return Halide::Float(32);
                case NPY_DOUBLE: return Halide::Float(64);
                case NPY_LONGDOUBLE: return Halide::Float(128);
                default: Halide::Handle();
            }
            return Halide::Handle();
        }
    
        NPY_TYPES for_nbits(int nbits, bool signed_type) {
            if (signed_type) {
                switch (nbits) {
                    case 1: return NPY_BOOL;
                    case 8: return NPY_INT8;
                    case 16: return NPY_INT16;
                    case 32: return NPY_INT32;
                }
            } else {
                switch (nbits) {
                    case 1: return NPY_BOOL;
                    case 8: return NPY_UINT8;
                    case 16: return NPY_UINT16;
                    case 32: return NPY_UINT32;
                }
            }
            return NPY_USERDEF;
        }
        
    }
    
    void PythonBufferImage::populate_buffer(Py_buffer* buffer, int flags) {
        buffer->buf = Image::rowp(0);
        buffer->format = ::strdup("b");
        buffer->ndim = (Py_ssize_t)Image::ndims();
        buffer->shape = new Py_ssize_t[Image::ndims()];
        buffer->strides = new Py_ssize_t[Image::ndims()];
        // buffer->suboffsets = new Py_ssize_t[Image::ndims()];
        buffer->suboffsets = NULL;
        
        int len = 1;
        for (int idx = 0; idx < buffer->ndim; idx++) {
            len *= Image::dim_or(idx, 1);
            buffer->shape[idx] = Image::dim_or(idx, 1);
            buffer->strides[idx] = Image::stride(idx);
            // buffer->suboffsets[idx] = Image:: ???
        }
        
        buffer->len = len;
        buffer->itemsize = Image::nbytes();
        
        buffer->readonly = true;
        // buffer->internal = NULL;
        buffer->obj = NULL;
    }
    
    HybridArray::HybridArray()
        :HalBase(), PythonBufferImage(), MetaImage()
        ,dtype_(NPY_UINT8)
        {}
    
    HybridArray::HybridArray(NPY_TYPES d, int x, int y, int z, int w, const std::string& name)
        :HalBase(detail::for_dtype(d), x, y, z, w, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HybridArray::HybridArray(NPY_TYPES d, int x, int y, int z, const std::string& name)
        :HalBase(detail::for_dtype(d), x, y, z, 0, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HybridArray::HybridArray(NPY_TYPES d, int x, int y, const std::string& name)
        :HalBase(detail::for_dtype(d), x, y, 0, 0, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HybridArray::HybridArray(NPY_TYPES d, int x, const std::string& name)
        :HalBase(detail::for_dtype(d), x, 0, 0, 0, name), PythonBufferImage(), MetaImage(name)
        ,dtype_(d)
        {}
    
    HybridArray::~HybridArray() {}
    
    PyObject* HybridArray::metadataPyObject() {
        const std::string& s = MetaImage::get_meta();
        if (s != "") { return PyBytes_FromString(s.c_str()); }
        Py_RETURN_NONE;
    }
    
    /// This returns the same type of data as buffer_t.host
    uint8_t* HybridArray::data(int s) const {
        return (uint8_t*)HalBase::buffer.host_ptr();
    }
    
    Halide::Type HybridArray::type() const {
        return detail::for_dtype(dtype_);
    }
    
    int HybridArray::nbits() const {
        return detail::for_dtype(dtype_).bits;
    }
    
    int HybridArray::nbytes() const {
        const int bits = detail::for_dtype(dtype_).bits;
        return (bits / 8) + bool(bits % 8);
    }
    
    int HybridArray::ndims() const {
        return HalBase::dimensions();
    }
    
    int HybridArray::dim(int d) const {
        return HalBase::extent(d);
    }
    
    int HybridArray::stride(int s) const {
        return HalBase::stride(s);
    }
    
    off_t HybridArray::rowp_stride() const {
        return HalBase::channels() == 1 ? 0 : off_t(HalBase::stride(1));
    }
    
    void* HybridArray::rowp(int r) const {
        uint8_t* host = data();
        host += off_t(r * rowp_stride());
        return static_cast<void*>(host);
    }
    
    /// extent, stride, min
    NPY_TYPES HybridArray::dtype()  { return dtype_; }
    int32_t* HybridArray::dims()    { return HalBase::raw_buffer()->extent; }
    int32_t* HybridArray::strides() { return HalBase::raw_buffer()->stride; }
    int32_t* HybridArray::offsets() { return HalBase::raw_buffer()->min; }
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    ArrayFactory::ArrayFactory()
        :nm("")
        {}
    
    ArrayFactory::ArrayFactory(const std::string& n)
        :nm(n)
        {}
    
    ArrayFactory::~ArrayFactory() {}
    
    std::string& ArrayFactory::name() { return nm; }
    void ArrayFactory::name(const std::string& n) { nm = n; }
    
    std::unique_ptr<Image> ArrayFactory::create(int nbits,
                                                int xHEIGHT, int xWIDTH, int xDEPTH,
                                                int d3, int d4) {
        return std::unique_ptr<Image>(
            new HybridArray(
                detail::for_nbits(nbits),
                xWIDTH, xHEIGHT, xDEPTH));
    }
    
    std::shared_ptr<Image> ArrayFactory::shared(int nbits,
                                                int xHEIGHT, int xWIDTH, int xDEPTH,
                                                int d3, int d4) {
        return std::shared_ptr<Image>(
            new HybridArray(
                detail::for_nbits(nbits),
                xWIDTH, xHEIGHT, xDEPTH));
    }
    
#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
    
    

}