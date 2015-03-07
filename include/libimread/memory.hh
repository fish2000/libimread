// Copyright 2013 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)
#ifndef LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013
#define LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013

#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {

    class memory_source : public byte_source {
        public:
            memory_source(const byte *c, const int l)
                :data(c), len(l), pos(0)
                { }
            
            ~memory_source() { }
            
            virtual std::size_t read(byte *buffer, std::size_t n) {
                if (pos + n > len) { n = len-pos; }
                std::memmove(buffer, data + pos, n);
                pos += n;
                return n;
            }
            
            virtual bool can_seek() const { return true; }
            virtual std::size_t seek_absolute(std::size_t p) { return pos = p; }
            virtual std::size_t seek_relative(int delta) { return pos += delta; }
            virtual std::size_t seek_end(int delta) { return pos = (len-delta-1); }
        
        private:
            const byte *data;
            const std::size_t len;
            std::size_t pos;
    };

}

#endif // LPC_MEMORY_H_INCLUDE_GUARD_MON_JUL__8_15_49_46_UTC_2013
