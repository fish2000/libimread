/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_H5DEETS_HH_
#define LIBIMREAD_EXT_H5DEETS_HH_

#include <string>
#include <functional>
#include <type_traits>
#include <libimread/libimread.hpp>

#include <H5Cpp.h>
#include <hdf5.h>

namespace im {
    
    namespace detail {
        
        struct h5base {
            
            using releaser_f = std::function<herr_t(hid_t)>;
            
            /// explicit hid+releaser construction:
            explicit h5base(hid_t, releaser_f);
            
            /// virtual destructor:
            virtual ~h5base();
            
            /// obtain the wrapped hid value:
            hid_t    get()   const;
            operator hid_t() const;
            
            /// call the wrapped releaser:
            herr_t release();
            
            /// boolean comparitors:
            bool operator==(h5base const&) const;
            bool operator!=(h5base const&) const;
            
            /// refcount shortcuts:
            int incref();
            int decref();
            
            protected:
                /// no-op and generic releaser functions:
                static const releaser_f unref;
                static const releaser_f NOOp;
            
            protected:
                /// hid and releaser values:
                hid_t m_hid = -1;
                releaser_f m_releaser = h5base::unref;
            
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
            explicit h5t_t(hid_t);
            explicit h5t_t(h5t_class_t, std::size_t);
            
            /// copy/move construct/assign:
            h5t_t(h5t_t const&);
            h5t_t(h5t_t&&) noexcept;
            h5t_t& operator=(h5t_t const&);
            h5t_t& operator=(h5t_t&&) noexcept;
            
            /// convenience getters:
            h5t_class_t cls() const;
            h5t_t super() const;
            
        };
        
        template <typename T> inline
        h5t_t typecode();
        
        #define DECLARE_TYPECODE(__typename__, __hdf5type__)                            \
            template <> inline                                                          \
            h5t_t typecode<__typename__>() { return h5t_t(__hdf5type__); }
        
        DECLARE_TYPECODE(char,                  H5T_NATIVE_CHAR);
        DECLARE_TYPECODE(signed char,           H5T_NATIVE_SCHAR);
        DECLARE_TYPECODE(byte,                  H5T_NATIVE_UCHAR);
        DECLARE_TYPECODE(short,                 H5T_NATIVE_SHORT);
        DECLARE_TYPECODE(unsigned short,        H5T_NATIVE_USHORT);
        DECLARE_TYPECODE(int,                   H5T_NATIVE_INT);
        DECLARE_TYPECODE(unsigned int,          H5T_NATIVE_UINT);
        DECLARE_TYPECODE(long,                  H5T_NATIVE_LONG);
        DECLARE_TYPECODE(unsigned long,         H5T_NATIVE_ULONG);
        DECLARE_TYPECODE(long long,             H5T_NATIVE_LLONG);
        DECLARE_TYPECODE(unsigned long long,    H5T_NATIVE_ULLONG);
        DECLARE_TYPECODE(float,                 H5T_NATIVE_FLOAT);
        DECLARE_TYPECODE(double,                H5T_NATIVE_DOUBLE);
        DECLARE_TYPECODE(long double,           H5T_NATIVE_LDOUBLE);
        DECLARE_TYPECODE(bool,                  H5T_NATIVE_HBOOL);
        
        #undef DECLARE_TYPECODE
        
        struct attspace_t : public h5base {
            
            /// construct empty wrapper:
            attspace_t(void);
            
            /// construct from hid and take ownership:
            explicit attspace_t(hid_t);
            
            /// create new dataspaces:
            static attspace_t scalar();
            static attspace_t simple();
            
            /// copy/move construct/assign:
            attspace_t(attspace_t const&);
            attspace_t(attspace_t&&) noexcept;
            attspace_t& operator=(attspace_t const&);
            attspace_t& operator=(attspace_t&&) noexcept;
            
        };
        
        struct h5a_t : public h5base {
            
            /// explicit by-index and by-name constructors:
            explicit h5a_t(hid_t parent_hid, std::size_t idx);
            explicit h5a_t(hid_t parent_hid, std::string const& name);
            
            /// explicit create-from-scratch constructor:
            explicit h5a_t(hid_t parent_hid, std::string const& name,
                           hid_t dataspace_hid,
                           h5t_t datatype);
            
            /// move constructor:
            h5a_t(h5a_t&& other) noexcept;
            
            /// virtual destructor:
            virtual ~h5a_t();
            
            /// API
            herr_t  read(void*) const;
            herr_t  read(void*, h5t_t const&) const;
            herr_t write(const void*);
            herr_t write(const void*, h5t_t const&);
            
            template <typename ToType>
            ToType typed_read() const {
                ToType value{};
                this->read(&value, typecode<
                           std::remove_cv_t<
                    std::remove_reference_t<ToType>>>());
                return value;
            }
            
            template <typename FromType>
            FromType typed_write(FromType&& value) {
                FromType out{ std::forward<FromType>(value) };
                this->write(&out, typecode<
                          std::remove_cv_t<
                   std::remove_reference_t<FromType>>>());
                return out;
            }
            
            hid_t parent() const;
            h5t_t const& memorytype() const;
            h5t_t const& memorytype(h5t_t const&);
            attspace_t dataspace() const;
            std::size_t idx() const;
            std::string name() const;
            
            protected:
                hid_t m_parent_hid = -1;
                h5t_t m_memorytype = h5t_t(H5T_NATIVE_INT);
                attspace_t m_dataspace;
                std::size_t m_idx = 0;
                mutable std::string m_name = NULL_STR;
            
            private:
                h5a_t(h5a_t const&);                /// NO COPYING!
                h5a_t& operator=(h5a_t const&);     /// OF ANY SORT!
        };
        
        
    } /* namespace detail */
    
}

#endif /// LIBIMREAD_EXT_H5DEETS_HH_