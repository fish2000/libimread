/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_H5DEETS_HH_
#define LIBIMREAD_EXT_H5DEETS_HH_

#include <functional>
#include <libimread/libimread.hpp>

#include <H5Cpp.h>
#include <hdf5.h>

namespace im {
    
    namespace detail {
        
        struct h5base {
            
            using releaser_f = std::function<herr_t(hid_t)>;
            
            /// explicit hid+releaser construction:
            explicit h5base(hid_t hid, releaser_f releaser);
            
            /// virtual destructor:
            virtual ~h5base();
            
            /// obtain the wrapped hid value:
            hid_t    get()   const;
            operator hid_t() const;
            
            /// call the wrapped releaser:
            herr_t release();
            
            protected:
                /// default no-op releaser function:
                static const releaser_f NOOp;
            
            protected:
                /// hid and releaser values:
                hid_t m_hid = -1;
                releaser_f m_releaser = h5base::NOOp;
            
            private:
                /// NO DEFAULT CONSTRUCTION FROM THE BASE TYPE:
                h5base(void);
                h5base(h5base const&);
                h5base(h5base&&);
                h5base& operator=(h5base const&);
                h5base& operator=(h5base&&);
            
        };
        
        struct h5t_t : public h5base {
            
            /// capitalization OCD:
            using h5t_class_t = H5T_class_t;
            
            /// explicit type and class constructors
            explicit h5t_t(hid_t hid);
            explicit h5t_t(h5t_class_t cls, std::size_t size);
            
            /// copy and move constructors: 
            h5t_t(h5t_t const& other);
            h5t_t(h5t_t&& other) noexcept;
            
            /// convenience getters:
            h5t_class_t cls() const;
            h5t_t super() const;
            
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
    
}

#endif /// LIBIMREAD_EXT_H5DEETS_HH_