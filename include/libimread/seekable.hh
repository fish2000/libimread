/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_SEEKABLE_HH_
#define LIBIMREAD_SEEKABLE_HH_

#include <inttypes.h>
#include <cassert>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <unistd.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>

namespace im {
    
    struct seekable {
        virtual ~seekable() { }
        virtual bool can_seek() const { return false; }
        virtual std::size_t seek_absolute(std::size_t) { throw NotImplementedError(); }
        virtual std::size_t seek_relative(int) { throw NotImplementedError(); }
        virtual std::size_t seek_end(int) { throw NotImplementedError(); }
    };
    
    class byte_source : virtual public seekable {
        
        public:
            virtual ~byte_source() { }
            virtual std::size_t read(byte *buffer, std::size_t) xWARN_UNUSED = 0;
            
            template <std::size_t Nelems>
            std::size_t read(byte (&arr)[Nelems], size_t n) {
                assert(n <= Nelems);
                byte *p = arr;
                return this->read(p, n);
            }
            
            void read_check(byte *buffer, std::size_t n) {
                if (this->read(buffer, n) != n) {
                    throw CannotReadError("SOURCE READ ERROR:",
                        "im::byte_source::read_check(): File ended prematurely");
                }
            }
            
            virtual std::vector<byte> full_data() {
                std::vector<byte> res;
                byte buffer[4096];
                while (std::size_t n = this->read(buffer, sizeof(buffer))) {
                    res.insert(res.end(), buffer, buffer + n);
                }
                return res;
            }
            
    };
    
    class byte_sink : virtual public seekable {
        public:
            virtual ~byte_sink() { }
            virtual std::size_t write(const void *buffer, std::size_t n) = 0;
            
            template <std::size_t Nelems>
            std::size_t write(byte (&arr)[Nelems], size_t n) {
                assert(n <= Nelems);
                byte *p = arr;
                return this->write(p, n);
            }
            
            void write_check(const byte *buffer, std::size_t n) {
                if (this->write(buffer, n) != n) {
                    throw CannotWriteError("SINK WRITE ERROR:",
                        "im::byte_sink::write_check(): Writing failed");
                }
            }
            virtual void flush() { }
            
            template <typename ...Args>
            std::size_t writef(const char *fmt, Args... args) {
                char buffer[1024];
                std::snprintf(buffer, 1024, fmt, args...);
                return this->write(buffer, std::strlen(static_cast<const char*>(buffer)));
            }
            
            virtual byte_sink& operator<<(const std::string &w) {
                this->write(static_cast<const char*>(w.c_str()),
                            static_cast<std::size_t>(w.length()));
                return *this;
            }
            virtual byte_sink& operator<<(const char *w) {
                this->write(w, std::strlen(w));
                return *this;
            }
            virtual byte_sink& operator<<(const std::vector<byte> &w) {
                this->write(&w[0], w.size());
                return *this;
            }
    };

}

#endif /// LIBIMREAD_SEEKABLE_HH_