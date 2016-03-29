/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_HYBRID_HH_
#define LIBIMREAD_PYTHON_HYBRID_HH_

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <vector>
#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>

namespace im {
    
    namespace detail {
        
        char const* structcode(NPY_TYPES dtype);
        Halide::Type for_dtype(NPY_TYPES dtype);
        NPY_TYPES for_nbits(int nbits, bool signed_type = false);
        
    }
    
    class PythonBufferImage : public Image {
        
        public:
            int populate_buffer(Py_buffer* buffer, NPY_TYPES dtype = NPY_UINT8,
                                                   int flags = 0);
            void release_buffer(Py_buffer* buffer);
        
    };
    
    /// We use Halide::ImageBase instead of Halide::Image here,
    /// so that we don't have to muck around with templates when
    /// working with arbitrary NumPy dtype values.
    using HalBase = Halide::ImageBase;
    using MetaImage = ImageWithMetadata;
    
    class HalideNumpyImage : public HalBase, public PythonBufferImage, public MetaImage {
        public:
            HalideNumpyImage();
            HalideNumpyImage(NPY_TYPES d, const buffer_t* b, std::string const& name="");
            HalideNumpyImage(NPY_TYPES d, int x, int y, int z, int w, std::string const& name="");
            HalideNumpyImage(NPY_TYPES d, int x, int y, int z, std::string const& name="");
            HalideNumpyImage(NPY_TYPES d, int x, int y, std::string const& name="");
            HalideNumpyImage(NPY_TYPES d, int x, std::string const& name="");
            
            using HalBase::dimensions;
            using HalBase::extent;
            using HalBase::stride;
            using HalBase::channels;
            using HalBase::buffer;
            using HalBase::raw_buffer;
            
            virtual ~HalideNumpyImage();
            PyObject* metadataPyObject();
            
            /// This returns the same type of data as buffer_t.host
            virtual uint8_t* data(int s = 0) const;
            
            Halide::Type type() const;
            buffer_t* buffer_ptr() const;
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
    
    class HybridFactory : public ImageFactory {
        
        private:
            std::string nm;
        
        public:
            using image_t = HalideNumpyImage;
            
            HybridFactory();
            HybridFactory(std::string const& n);
            
            virtual ~HybridFactory();
            std::string& name();
            void name(std::string const& n);
        
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

#endif /// LIBIMREAD_PYTHON_HYBRID_HH_
