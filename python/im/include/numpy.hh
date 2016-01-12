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

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>

// #if PY_MAJOR_VERSION < 3
// #define PyBytes_FromString(string) PyString_FromString(string)
// #endif

namespace im {
    
    namespace detail {
        
        Halide::Type for_dtype(NPY_TYPES dtype);
        NPY_TYPES for_nbits(int nbits, bool signed_type = false);
        
    }
    
    class PythonBufferImage : public Image {
        
        void populate_buffer(Py_buffer* buffer, int flags = 0);
        
    };
    
    /// We use Halide::ImageBase instead of Halide::Image here,
    /// so that we don't have to muck around with templates when
    /// working with arbitrary NumPy dtype values.
    using HalBase = Halide::ImageBase;
    using MetaImage = ImageWithMetadata;
    
    class HybridArray : public HalBase, public PythonBufferImage, public MetaImage {
        public:
            HybridArray();
            HybridArray(NPY_TYPES d, int x, int y, int z, int w, const std::string& name="");
            HybridArray(NPY_TYPES d, int x, int y, int z, const std::string& name="");
            HybridArray(NPY_TYPES d, int x, int y, const std::string& name="");
            HybridArray(NPY_TYPES d, int x, const std::string& name="");
            
            using HalBase::dimensions;
            using HalBase::extent;
            using HalBase::stride;
            using HalBase::channels;
            using HalBase::buffer;
            using HalBase::raw_buffer;
            
            virtual ~HybridArray();
            PyObject* metadataPyObject();
            
            /// This returns the same type of data as buffer_t.host
            virtual uint8_t* data(int s = 0) const;
            
            inline Halide::Type type() const;
            virtual int nbits() const override;
            virtual int nbytes() const override;
            virtual int ndims() const override;
            virtual int dim(int d) const override;
            virtual int stride(int s) const override;
            inline off_t rowp_stride() const;
            virtual void* rowp(int r) const override;
            
            /// extent, stride, min
            virtual NPY_TYPES dtype();
            virtual int32_t* dims();
            virtual int32_t* strides();
            virtual int32_t* offsets();
            
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
            ArrayFactory();
            ArrayFactory(const std::string& n);
            
            virtual ~ArrayFactory();
            std::string& name();
            void name(const std::string& n);
        
        protected:
            virtual std::unique_ptr<Image> create(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) override;
            
            virtual std::shared_ptr<Image> shared(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) override;
    };

#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH

}

#endif /// LIBIMREAD_PYTHON_NUMPY_HH_
