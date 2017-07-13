/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_HYBRID_HH_
#define LIBIMREAD_PYTHON_HYBRID_HH_

#include <cstring>
#include <vector>
#include <memory>
#include <string>

#include <Python.h>
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <Halide.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/image.hh>
#include <libimread/metadata.hh>
#include <libimread/options.hh>
#include "check.hh"

#ifdef __CPP1z__
#include <string_view>
#else
namespace std {
    using string_view = std::string; /// WTF HAX
}
#endif

namespace im {
    
    enum class Endian : char {
        Unspecified = '|',
        Little      = '<',
        Big         = '>'
    };
    
    extern const Endian byteorder;
    
    namespace detail {
        
        char const* structcode(NPY_TYPES dtype);
        halide_type_t for_dtype(NPY_TYPES dtype);
        NPY_TYPES for_nbits(int nbits, bool signed_type = false);
        
        template <typename PixelType>
        struct encode_type;
        
        #define DEFINE_TYPE_ENCODING(PixelType, character)                                      \
        template <>                                                                             \
        struct encode_type<PixelType> {                                                         \
            using type = PixelType;                                                             \
            static constexpr int typesize() { return sizeof(type); }                            \
            static constexpr char typekind() { return character; }                              \
            static std::string typestr(Endian endian = Endian::Unspecified) {                   \
                char endianness = static_cast<char>(endian);                                    \
                int buffer_size = std::snprintf(nullptr, 0,                                     \
                                                "%c%c%lu",                                      \
                                                endianness, typekind(), sizeof(type)) + 1;      \
                char out_buffer[buffer_size];                                                   \
                __attribute__((unused))                                                         \
                int buffer_used = std::snprintf(out_buffer, buffer_size,                        \
                                                "%c%c%lu",                                      \
                                                endianness, typekind(), sizeof(type));          \
                return std::string(out_buffer);                                                 \
            }                                                                                   \
        };
        
        DEFINE_TYPE_ENCODING(bool,           'b');
        DEFINE_TYPE_ENCODING(uint8_t,        'u');
        DEFINE_TYPE_ENCODING(uint16_t,       'u');
        DEFINE_TYPE_ENCODING(uint32_t,       'u');
        DEFINE_TYPE_ENCODING(uint64_t,       'u');
        DEFINE_TYPE_ENCODING(int8_t,         'i');
        DEFINE_TYPE_ENCODING(int16_t,        'i');
        DEFINE_TYPE_ENCODING(int32_t,        'i');
        DEFINE_TYPE_ENCODING(int64_t,        'i');
        DEFINE_TYPE_ENCODING(float,          'f');
        DEFINE_TYPE_ENCODING(double,         'f');
        DEFINE_TYPE_ENCODING(long double,    'f');
        DEFINE_TYPE_ENCODING(PyObject*,      'O');
        DEFINE_TYPE_ENCODING(char*,          'S');
        DEFINE_TYPE_ENCODING(char const*,    'S');
        DEFINE_TYPE_ENCODING(wchar_t*,       'U');
        
        template <typename PointerBase>
        struct encode_type<PointerBase*> {
            using type = std::add_pointer_t<PointerBase>;
            static constexpr int typesize() { return sizeof(type); }
            static constexpr char typekind() { return 'V'; }
            static std::string typestr(Endian e = Endian::Unspecified) {
                char endianness = static_cast<char>(e);
                int buffer_size = std::snprintf(nullptr, 0,
                                                "%cV%i", endianness, typesize()) + 1;
                char out_buffer[buffer_size];
                __attribute__((unused))
                int buffer_used = std::snprintf(out_buffer, buffer_size,
                                                "%cV%i", endianness, typesize());
                return std::string(out_buffer);
            }
        };
        
        template <typename PixelType> inline
        std::string encoding_for(Endian e = Endian::Unspecified) {
            return encode_type<std::remove_cv_t<
                               std::decay_t<PixelType>>>::typestr(e);
        }
        
        template <typename PixelType> inline
        const char character_for() {
            return encode_type<std::remove_cv_t<
                               std::decay_t<PixelType>>>::typekind();
        }
        
    }; /* namespace detail */
    
    class PythonBufferImage : public Image {
        
        public:
            int populate_buffer(Py_buffer* buffer, NPY_TYPES dtype = NPY_UINT8,
                                                   int flags = 0);
            void release_buffer(Py_buffer* buffer);
        
    };
    
//     /// We use Halide::ImageBase instead of Halide::Image here,
//     /// so that we don't have to muck around with templates when
//     /// working with arbitrary NumPy dtype values.
//     using HalBase = Halide::Runtime::Buffer<>;
    using MetaImage = ImageWithMetadata;
//
//     /// forward-declare factory class
//     class HybridFactory;
//
//     class HalideNumpyImage : public HalBase, public PythonBufferImage, public MetaImage {
//
//         public:
//             using factory_t = HybridFactory;
//
//             using HalBase::dimensions;
//             using HalBase::extent;
//             using HalBase::stride;
//             using HalBase::channels;
//             using HalBase::raw_buffer;
//
//             HalideNumpyImage();
//             HalideNumpyImage(NPY_TYPES d, buffer_t const* b,            std::string const& name = "");
//             HalideNumpyImage(NPY_TYPES d, int x, int y, int z, int w,   std::string const& name = "");
//             HalideNumpyImage(NPY_TYPES d, int x, int y, int z,          std::string const& name = "");
//             HalideNumpyImage(NPY_TYPES d, int x, int y,                 std::string const& name = "");
//             HalideNumpyImage(NPY_TYPES d, int x,                        std::string const& name = "");
//
//             explicit HalideNumpyImage(HalideNumpyImage const& other,
//                                       int const zidx = 0,               std::string const& name = "");
//             explicit HalideNumpyImage(HalideNumpyImage const& basis,
//                                       HalideNumpyImage const& etc,      std::string const& name = "");
//
//             virtual ~HalideNumpyImage();
//
//             /// This returns the same type of data as buffer_t.host
//             virtual uint8_t* data() const;
//             virtual uint8_t* data(int s) const;
//             virtual std::string_view view() const;
//
//             halide_type_t type() const;
//             buffer_t* buffer_ptr() const;
//             virtual int nbits() const override;
//             virtual int nbytes() const override;
//             virtual int ndims() const override;
//             virtual int dim(int d) const override;
//             virtual int stride(int s) const override;
//             virtual int min(int s) const override;
//             virtual bool is_signed() const override;
//             virtual bool is_floating_point() const override;
//             inline off_t rowp_stride() const;
//             virtual void* rowp(int r) const override;
//
//             /// type encoding
//             virtual NPY_TYPES dtype() const;
//             virtual char dtypechar() const;
//             virtual std::string dtypename() const;
//             virtual char const* structcode() const;
//             virtual std::string dsignature(Endian e = Endian::Unspecified) const;
//
//             /// extent, stride, min
//             virtual int32_t* dims();
//             virtual int32_t* strides();
//             virtual int32_t* offsets();
//
//         private:
//             NPY_TYPES dtype_;
//     };
//
// #define xWIDTH d1
// #define xHEIGHT d0
// #define xDEPTH d2
//
//     class HybridFactory : public ImageFactory {
//
//         private:
//             std::string nm;
//
//         public:
//             using image_t = HalideNumpyImage;
//
//             HybridFactory();
//             HybridFactory(std::string const& n);
//
//             virtual ~HybridFactory();
//             std::string& name();
//             void name(std::string const& n);
//
//             static PyTypeObject* image_type() { return &ImageModel_Type; }
//             static PyTypeObject* buffer_type() { return &ImageBufferModel_Type; }
//
//         protected:
//             virtual std::unique_ptr<Image> create(int nbits,
//                                           int xHEIGHT, int xWIDTH, int xDEPTH,
//                                           int d3, int d4) override;
//     };
//
// #undef xWIDTH
// #undef xHEIGHT
// #undef xDEPTH
    
    /// forward-declare factory class
    class ArrayFactory;
    
    class ArrayImage : public PythonBufferImage, public MetaImage {
        
        public:
            using factory_t = ArrayFactory;
            
            ArrayImage();
            ArrayImage(NPY_TYPES d, buffer_t const* b,            std::string const& name = "");
            ArrayImage(NPY_TYPES d, int x, int y, int z, int w,   std::string const& name = "");
            ArrayImage(NPY_TYPES d, int x, int y, int z,          std::string const& name = "");
            ArrayImage(NPY_TYPES d, int x, int y,                 std::string const& name = "");
            ArrayImage(NPY_TYPES d, int x,                        std::string const& name = "");
            
            ArrayImage(ArrayImage const& other);
            ArrayImage(ArrayImage&& other) noexcept;
            
            explicit ArrayImage(ArrayImage const& other,
                                int const zidx = 0,               std::string const& name = "");
            explicit ArrayImage(ArrayImage const& basis,
                                ArrayImage const& etc,            std::string const& name = "");
            
            virtual ~ArrayImage();
            
            /// This returns the same type of data as buffer_t.host
            virtual uint8_t* data() const;
            virtual uint8_t* data(int s) const;
            virtual std::string_view view() const;
            
            halide_type_t type() const;
            buffer_t* buffer_ptr() const;
            virtual int nbits() const override;
            virtual int nbytes() const override;
            virtual int ndims() const override;
            virtual int dim(int d) const override;
            virtual int stride(int s) const override;
            virtual int min(int s) const override;
            virtual bool is_signed() const override;
            virtual bool is_floating_point() const override;
            virtual void* rowp(int r) const override;
            
            /// type encoding
            virtual NPY_TYPES dtype() const;
            virtual char dtypechar() const;
            virtual std::string dtypename() const;
            virtual char const* structcode() const;
            virtual std::string dsignature(Endian e = Endian::Unspecified) const;
            
            /// extent, stride, min
            virtual int32_t* dims();
            virtual int32_t* strides();
            virtual int32_t* offsets();
            
        private:
            PyArrayObject* array;
            buffer_t* buffer;
            bool deallocate = false;
    };
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    class ArrayFactory : public ImageFactory {
        
        private:
            std::string nm;
        
        public:
            using image_t = ArrayImage;
            
            ArrayFactory();
            ArrayFactory(std::string const& n);
            
            virtual ~ArrayFactory();
            std::string& name();
            void name(std::string const& n);
            
            static PyTypeObject* image_type() { return &ArrayModel_Type; }
            static PyTypeObject* buffer_type() { return &ArrayBufferModel_Type; }
        
        protected:
            virtual std::unique_ptr<Image> create(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) override;
    };

#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH

}

#endif /// LIBIMREAD_PYTHON_HYBRID_HH_
