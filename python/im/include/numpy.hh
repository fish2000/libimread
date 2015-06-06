/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_NUMPY_HH_
#define LIBIMREAD_PYTHON_NUMPY_HH_

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <cstring>
#include <vector>
#include <memory>
#include <sstream>

#include "typecode.hpp"

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>

#if PY_MAJOR_VERSION < 3
#define PyBytes_FromString(string) PyString_FromString(string)
#endif

namespace im {
    
    /*
    constexpr Halide::Type for_nbits(int nbits) {
        switch (nbits) {
            case 1: return Halide::Bool();
            case 8: return Halide::UInt(8);
            case 16: return Halide::UInt(16);
            case 32: return Halide::UInt(32);
        }
        return Halide::Handle();
    }
    */
    
    constexpr Halide::Type for_dtype(NPY_TYPES dtype) {
        switch (dtype) {
            case NPY_BOOL: return Halide::Bool();
            case NPY_UINT8: return Halide::UInt(8);
            case NPY_UINT16: return Halide::UInt(16);
            case NPY_UINT32: return Halide::UInt(32);
        }
        return Halide::Handle();
    }
    
    constexpr NPY_TYPES for_nbits(int nbits) {
        switch (nbits) {
            case 1: return NPY_BOOL;
            case 8: return NPY_UINT8;
            case 16: return NPY_UINT16;
            case 32: return NPY_UINT32;
        }
        return NPY_USERDEF;
    }
    
    /// We use Halide::ImageBase instead of Halide::Image here,
    /// so that we don't have to muck around with templates when
    /// working with arbitrary NumPy dtype values.
    using HalBase = Halide::ImageBase;
    using MetaImage = ImageWithMetadata;
    
    class HybridArray : public HalBase, public Image, public MetaImage {
        public:
            HybridArray()
                :HalBase(), Image(), MetaImage()
                ;dtype_(NPY_UINT8)
                {}
            
            HybridArray(NPY_TYPES d, int x, int y, int z, int w, const std::string &name="")
                :HalBase(for_dtype(d), x, y, z, w, name), Image(), MetaImage(name)
                ;dtype_(d)
                {}
            
            HybridArray(NPY_TYPES d, int x, int y, int z, const std::string &name="")
                :HalBase(for_dtype(d), x, y, z, name), Image(), MetaImage(name)
                ;dtype_(d)
                {}
            
            HybridArray(NPY_TYPES d, int x, int y, const std::string &name="")
                :HalBase(for_dtype(d), x, y, name), Image(), MetaImage(name)
                ;dtype_(d)
                {}
            
            HybridArray(NPY_TYPES d, int x, const std::string &name="")
                :HalBase(for_dtype(d), x, name), Image(), MetaImage(name)
                ;dtype_(d)
                {}
            
            using HalBase::dimensions;
            using HalBase::extent;
            using HalBase::stride;
            using HalBase::channels;
            using HalBase::buffer;
            using HalBase::raw_buffer;
            
            virtual ~HybridArray() {}
            
            PyObject *metadataPyObject() {
                std::string *s = MetaImage::get_meta();
                if (s) { return PyBytes_FromString(s->c_str()); }
                Py_RETURN_NONE;
            }
            
            /// This returns the same type of data as buffer_t.host
            virtual uint8_t data(int s) const {
                return HalBase::buffer.host_ptr();
            }
            
            inline constexpr Halide::Type type() const {
                return for_dtype(dtype_);
            }
            
            virtual int nbits() const override {
                return for_dtype(dtype_).bits;
            }
            
            virtual int nbytes() const override {
                const int bits = for_dtype(dtype_).bits;
                return (bits / 8) + bool(bits % 8);
            }
            
            virtual int ndims() const override {
                return HalBase::dimensions();
            }
            
            virtual int dim(int d) const override {
                return HalBase::extent(d);
            }
            
            virtual int stride(int s) const override {
                return HalBase::stride(s);
            }
            
            inline off_t rowp_stride() const {
                return HalBase::channels() == 1 ? 0 : off_t(HalBase::stride(1));
            }
            
            virtual void *rowp(int r) override {
                uint8_t *host = data();
                host += off_t(r * rowp_stride());
                return static_cast<void *>(host);
            }
            
            /// extent, stride, min
            virtual NPY_TYPES dtype()  { return dtype_; }
            virtual int32_t *dims()    { return HalBase::raw_buffer()->extent; }
            virtual int32_t *strides() { return HalBase::raw_buffer()->stride; }
            virtual int32_t *offsets() { return HalBase::raw_buffer()->min; }
            
        private:
            NPY_TYPES dtype_;
    };
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    class ArrayFactory : public ImageFactory {
        private:
            std::string nm;
        
        public:
            ArrayFactory()
                :nm(std::string(""))
                {}
            ArrayFactory(const std::string &n)
                :nm(std::string(n))
                {}
            
            virtual ~ArrayFactory() {}
            
            std::string &name() { return nm; }
            void name(std::string &nnm) { nm = nnm; }
        
        protected:
            virtual std::unique_ptr<Image> create(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) override {
                return std::unique_ptr<Image>(
                    new HybridArray(
                        for_nbits(nbits),
                        xWIDTH, xHEIGHT, xDEPTH));
            }
            
            virtual std::shared_ptr<Image> shared(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) override {
                return std::shared_ptr<Image>(
                    new HybridArray(
                        for_nbits(nbits),
                        xWIDTH, xHEIGHT, xDEPTH));
            }
    };

#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH

}

#endif /// LIBIMREAD_PYTHON_NUMPY_HH_
