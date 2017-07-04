/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_HDF5_HH_
#define LIBIMREAD_IO_HDF5_HH_

#include <H5Cpp.h>
#include <hdf5.h>

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    namespace detail {
        
        template <typename T> inline
        hid_t typecode();
        
        template <> inline
        hid_t typecode<char>() { return H5T_NATIVE_CHAR; }
        
        template <> inline
        hid_t typecode<signed char>() { return H5T_NATIVE_SCHAR; }
        
        template <> inline
        hid_t typecode<byte>() { return H5T_NATIVE_UCHAR; }
        
        template <> inline
        hid_t typecode<short>() { return H5T_NATIVE_SHORT; }
        
        template <> inline
        hid_t typecode<unsigned short>() { return H5T_NATIVE_USHORT; }
        
        template <> inline
        hid_t typecode<int>() { return H5T_NATIVE_INT; }
        
        template <> inline
        hid_t typecode<unsigned int>() { return H5T_NATIVE_UINT; }
        
        template <> inline
        hid_t typecode<long>() { return H5T_NATIVE_LONG; }
        
        template <> inline
        hid_t typecode<unsigned long>() { return H5T_NATIVE_ULONG; }
        
        template <> inline
        hid_t typecode<long long>() { return H5T_NATIVE_LLONG; }
        
        template <> inline
        hid_t typecode<unsigned long long>() { return H5T_NATIVE_ULLONG; }
        
        template <> inline
        hid_t typecode<float>() { return H5T_NATIVE_FLOAT; }
        
        template <> inline
        hid_t typecode<double>() { return H5T_NATIVE_DOUBLE; }
        
        template <> inline
        hid_t typecode<long double>() { return H5T_NATIVE_LDOUBLE; }
        
        template <> inline
        hid_t typecode<bool>() { return H5T_NATIVE_HBOOL; }
        
        // template <> inline
        // hid_t typecode<CFTypeRef>() { return H5T_NATIVE_OPAQUE; }
        
    }
    
    
    class HDF5Format : public ImageFormatBase<HDF5Format> {
        
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            /// \x0d\x0a\x1a\x0a
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x89\x48\x44\x46", 4)
                },
                _suffixes = { "hdf5", "h5", "hdf" },
                _mimetype = "image/hdf5"
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override;
            
            virtual void write(Image& input,
                               byte_sink* output,
                               options_map const& opts) override;
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
