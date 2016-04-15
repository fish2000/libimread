/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_HDF5_HH_
#define LIBIMREAD_IO_HDF5_HH_

#include <hdf5.h>
#include <H5Cpp.h>
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
        
        // template <> inline
        // dtype type<CFTypeRef>() { return dtype::NATIVE_OPAQUE; }
        
    }
    
    
    class HDF5Format : public ImageFormatBase<HDF5Format> {
        
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            /// \x0d\x0a\x1a\x0a
            
            DECLARE_OPTIONS(
                base64::encode("\x89\x48\x44\x46", 4), /// [0]
                "hdf5",
                "image/hdf5");
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                const options_map& opts) override;
            
            virtual void write(Image& input,
                               byte_sink* output,
                               const options_map& opts) override;
    };
    
    namespace format {
        using H5 = HDF5Format;
        using HDF5 = HDF5Format;
    }
    
}

/*
 * Via: http://www.digitalpreservation.gov/formats/fdd/fdd000229.shtml
 *  ... which that looks like a useful resource in general
 */

#endif /// LIBIMREAD_IO_HDF5_HH_
