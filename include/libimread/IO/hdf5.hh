/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_HDF5_HH_
#define LIBIMREAD_IO_HDF5_HH_

#include <H5Cpp.h>
#import <CoreFoundation/CoreFoundation.h>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/options.hh>

namespace im {
    
    namespace detail {
        
        using dtype = H5::PredType;
        
        template <typename T> inline
        dtype type();
        
        template <> inline
        dtype type<char>() { return dtype::NATIVE_CHAR; }
        
        template <> inline
        dtype type<signed char>() { return dtype::NATIVE_SCHAR; }
        
        template <> inline
        dtype type<unsigned char>() { return dtype::NATIVE_UCHAR; }
        
        template <> inline
        dtype type<short>() { return dtype::NATIVE_SHORT; }
        
        template <> inline
        dtype type<unsigned short>() { return dtype::NATIVE_USHORT; }
        
        template <> inline
        dtype type<int>() { return dtype::NATIVE_INT; }
        
        template <> inline
        dtype type<unsigned int>() { return dtype::NATIVE_UINT; }
        
        template <> inline
        dtype type<long>() { return dtype::NATIVE_LONG; }
        
        template <> inline
        dtype type<unsigned long>() { return dtype::NATIVE_ULONG; }
        
        template <> inline
        dtype type<long long>() { return dtype::NATIVE_LLONG; }
        
        template <> inline
        dtype type<unsigned long long>() { return dtype::NATIVE_ULLONG; }
        
        template <> inline
        dtype type<float>() { return dtype::NATIVE_FLOAT; }
        
        template <> inline
        dtype type<double>() { return dtype::NATIVE_DOUBLE; }
        
        template <> inline
        dtype type<long double>() { return dtype::NATIVE_LDOUBLE; }
        
        template <> inline
        dtype type<bool>() { return dtype::NATIVE_HBOOL; }
        
        template <> inline
        dtype type<CFTypeRef>() { return dtype::NATIVE_OPAQUE; }
        
        /*
        template <> inline
        dtype type<char>() { return dtype::NATIVE_B8; }
        template <> inline
        dtype type<char>() { return dtype::NATIVE_B16; }
        template <> inline
        dtype type<char>() { return dtype::NATIVE_B32; }
        template <> inline
        dtype type<char>() { return dtype::NATIVE_B64; }
        */
        
        /*
        template <> inline
        dtype type<std::size_t>() { return dtype::NATIVE_HSIZE; }
        
        template <> inline
        dtype type<ssize_t>() { return dtype::NATIVE_HSSIZE; }
        
        template <> inline
        dtype type<errno_t>() { return dtype::NATIVE_HERR; }
        
        template <> inline
        dtype type<int8_t>() { return dtype::NATIVE_INT8; }
        
        template <> inline
        dtype type<uint8_t>() { return dtype::NATIVE_UINT8; }
        
        template <> inline
        dtype type<int16_t>() { return dtype::NATIVE_INT16; }
        
        template <> inline
        dtype type<uint16_t>() { return dtype::NATIVE_UINT16; }
        
        template <> inline
        dtype type<int32_t>() { return dtype::NATIVE_INT32; }
        
        template <> inline
        dtype type<uint32_t>() { return dtype::NATIVE_UINT32; }
        
        template <> inline
        dtype type<int64_t>() { return dtype::NATIVE_INT64; }
        
        template <> inline
        dtype type<uint64_t>() { return dtype::NATIVE_UINT64; }
        */
        
    }
    
    
    class HDF5Format : public ImageFormat {
        
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            DECLARE_OPTIONS(
                "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A",
                "hdf5",
                "image/hdf5");
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
    };
    
    namespace format {
        using H5 = HDF5Format;
        using HDF5 = HDF5Format;
    }
    
}


#endif /// LIBIMREAD_IO_HDF5_HH_
