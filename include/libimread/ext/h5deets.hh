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
            explicit h5base(hid_t, releaser_f);
            
            /// virtual destructor:
            virtual ~h5base();
            
            /// obtain the wrapped hid value:
            hid_t    get()   const;
            operator hid_t() const;
            
            /// call the wrapped releaser:
            herr_t release();
            
            bool operator==(h5base const&) const;
            bool operator!=(h5base const&) const;
            
            protected:
                /// no-op and generic releaser functions:
                static const releaser_f deref;
                static const releaser_f NOOp;
            
            protected:
                /// hid and releaser values:
                hid_t m_hid = -1;
                releaser_f m_releaser = h5base::deref;
            
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
            herr_t read(void*) const;
            herr_t read(void*, h5t_t const&) const;
            herr_t write(const void*);
            herr_t write(const void*, h5t_t const&);
            
            template <typename ToType>
            ToType typed_read() const {
                ToType value{};
                this->read(&value, typecode<ToType>());
                return value;
            }
            
            template <typename FromType>
            void typed_write(FromType&& value) {
                this->write(&std::forward<FromType>(value),
                                 typecode<FromType>());
            }
            
            h5t_t const& memorytype() const;
            h5t_t const& memorytype(h5t_t const&);
            hid_t parent() const;
            hid_t dataspace() const;
            std::size_t idx() const;
            std::string name() const;
            // std::string name(std::string const&);
            
            protected:
                h5t_t m_memorytype = h5t_t(H5T_NATIVE_UCHAR);
                hid_t m_parent_hid = -1;
                hid_t m_dataspace_hid = -1;
                std::size_t m_idx = 0;
                mutable std::string m_name = NULL_STR;
            
            private:
                h5a_t(h5a_t const&);                /// NO COPYING!
                h5a_t& operator=(h5a_t const&);     /// OF ANY SORT!
        };
        
        
    } /* namespace detail */
    
}

#endif /// LIBIMREAD_EXT_H5DEETS_HH_