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
        
        struct h5base {
            
            using releaser_f = std::function<herr_t(hid_t)>;
            
            explicit h5base(hid_t hid, releaser_f releaser)
                :m_hid(hid)
                ,m_releaser(releaser)
                {}
            
            virtual ~h5base() {
                /// call m_releaser on member HID:
                if (m_hid > 0) { m_releaser(m_hid); }
            }
            
            hid_t    get()   const { return m_hid; }
            operator hid_t() const { return m_hid; }
            
            herr_t release() const {
                if (m_hid < 0) { return -1; }
                herr_t out = m_releaser(m_hid);
                if (out > 0) { m_hid = -1; }
                return out;
            }
            
            protected:
                static const releaser_f NOOp;
            
            protected:
                hid_t m_hid = -1;
                releaser_f m_releaser = h5base::NOOp;
            
            private:
                h5base(void);
                h5base(h5base const&);
                h5base(h5base&&);
                h5base& operator=(h5base const&);
                h5base& operator=(h5base&&);
            
        };
        
        struct h5t_t : public h5base {
            
            /// capitalization OCD:
            using h5t_class_t = H5T_class_t;
            
            explicit h5t_t(hid_t hid)
                :h5base(H5Tcopy(hid),
                        H5Tclose)
                {}
            
            explicit h5t_t(h5t_class_t cls, std::size_t size)
                :h5base(H5Tcreate(cls, size),
                        H5Tclose)
                {}
            
            h5t_t(h5t_t const& other)
                :h5base(H5Tcopy(other.m_hid),
                                other.m_releaser)
                {}
            
            h5t_t(h5t_t&& other) noexcept
                :h5base(std::move(other.m_hid),
                        std::move(other.m_releaser))
                { other.m_hid = -1; }
            
            h5t_class_t cls() const {
                return H5Tget_class(m_hid);
            }
            
            h5t_t super() const {
                return h5t_t(H5Tget_super(m_hid));
            }
            
        };
        
        template <typename T> inline
        h5t_t typecode();
        
        template <> inline
        h5t_t typecode<char>() {
            return h5t_t(H5T_NATIVE_CHAR);
        }
        
        template <> inline
        h5t_t typecode<signed char>() {
            return h5t_t(H5T_NATIVE_SCHAR);
        }
        
        template <> inline
        h5t_t typecode<byte>() {
            return h5t_t(H5T_NATIVE_UCHAR);
        }
        
        template <> inline
        h5t_t typecode<short>() {
            return h5t_t(H5T_NATIVE_SHORT);
        }
        
        template <> inline
        h5t_t typecode<unsigned short>() {
            return h5t_t(H5T_NATIVE_USHORT);
        }
        
        template <> inline
        h5t_t typecode<int>() {
            return h5t_t(H5T_NATIVE_INT);
        }
        
        template <> inline
        h5t_t typecode<unsigned int>() {
            return h5t_t(H5T_NATIVE_UINT);
        }
        
        template <> inline
        h5t_t typecode<long>() {
            return h5t_t(H5T_NATIVE_LONG);
        }
        
        template <> inline
        h5t_t typecode<unsigned long>() {
            return h5t_t(H5T_NATIVE_ULONG);
        }
        
        template <> inline
        h5t_t typecode<long long>() {
            return h5t_t(H5T_NATIVE_LLONG);
        }
        
        template <> inline
        h5t_t typecode<unsigned long long>() {
            return h5t_t(H5T_NATIVE_ULLONG);
        }
        
        template <> inline
        h5t_t typecode<float>() {
            return h5t_t(H5T_NATIVE_FLOAT);
        }
        
        template <> inline
        h5t_t typecode<double>() {
            return h5t_t(H5T_NATIVE_DOUBLE);
        }
        
        template <> inline
        h5t_t typecode<long double>() {
            return h5t_t(H5T_NATIVE_LDOUBLE);
        }
        
        template <> inline
        h5t_t typecode<bool>() {
            return h5t_t(H5T_NATIVE_HBOOL);
        }
        
    } /* namespace detail */
    
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
