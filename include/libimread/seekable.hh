/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_SEEKABLE_HH_
#define LIBIMREAD_SEEKABLE_HH_

#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <unistd.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>

namespace im {
    
    struct seekable {
        virtual ~seekable() { }
        virtual bool can_seek() const noexcept { return false; }
        virtual std::size_t seek_absolute(std::size_t) { imread_raise_default(NotImplementedError); }
        virtual std::size_t seek_relative(int) { imread_raise_default(NotImplementedError); }
        virtual std::size_t seek_end(int) { imread_raise_default(NotImplementedError); }
    };
    
    class byte_source : virtual public seekable {
        
        public:
            virtual ~byte_source() { }
            virtual std::size_t read(byte* buffer, std::size_t) warn_unused = 0;
            
            template <std::size_t Nelems>
            std::size_t read(byte (&arr)[Nelems], std::size_t n) {
                imread_assert(n <= Nelems,
                    "write<Nelems>() called with n-value > template-value:",
                        FF("\t     n = %i", n),
                        FF("\tNelems = %i", Nelems));
                byte *p = arr;
                return this->read(p, n);
            }
            
            void read_check(byte* buffer, std::size_t n) {
                if (this->read(buffer, n) != n) {
                    imread_raise(CannotReadError, "File ended prematurely");
                }
            }
            
            virtual std::vector<byte> full_data() {
                std::vector<byte> res;
                std::size_t n;
                byte buffer[4096];
                while ((n = this->read(buffer, sizeof(buffer)))) {
                    res.insert(res.end(), buffer, buffer + n);
                }
                return res;
            }
            
    };
    
    class byte_sink : virtual public seekable {
        
        public:
            virtual ~byte_sink() { }
            virtual std::size_t write(const void* buffer, std::size_t n) = 0;
            
            template <std::size_t Nelems>
            std::size_t write(byte (&arr)[Nelems], std::size_t n) {
                imread_assert(n <= Nelems,
                    "write<Nelems>() called with n-value > template-value:",
                        FF("\t     n = %i", n),
                        FF("\tNelems = %i", Nelems));
                byte* p = arr;
                return this->write(p, n);
            }
            
            void write_check(const byte* buffer, std::size_t n) {
                std::size_t out = this->write(buffer, n);
                imread_assert(out == n,
                    "write_check() return value differs from n:",
                        FF("\t  n = %i", n),
                        FF("\tout = %i", out));
            }
            
            virtual void flush() { }
            
            template <typename ...Args>
            std::size_t writef(const char* fmt, Args... args) {
                char buffer[1024];
                // std::size_t buffer_size = std::snprintf(buffer, 1024, fmt, args...);
                std::snprintf(buffer, 1024, fmt, args...);
                return this->write(buffer, std::strlen(static_cast<const char*>(buffer)));
            }
    };

}

#endif /// LIBIMREAD_SEEKABLE_HH_