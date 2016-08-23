/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/memory.hh>

namespace im {
    
    memory_source::memory_source(const byte* c, const int len)
        :data(c), length(len), pos(0)
        {}
    
    memory_source::~memory_source() {}
    
    std::size_t memory_source::read(byte* buffer, std::size_t n) {
        if (pos + n > length) { n = length-pos; }
        /// FYI, std::memmove() actually copies bytes, rather
        /// than 'moving' them (whatever that might mean)
        std::memmove(buffer, data + pos, n);
        pos += n;
        return n;
    }
    
    bool memory_source::can_seek() const noexcept { return true; }
    std::size_t memory_source::seek_absolute(std::size_t p) { return pos = p; }
    std::size_t memory_source::seek_relative(int delta) { return pos += delta; }
    std::size_t memory_source::seek_end(int delta) { return pos = (length-delta-1); }
    
    std::vector<byte> memory_source::full_data() {
        std::vector<byte> result(length);
        std::memcpy(&result[0], data, length);
        return result;
    }
    
    std::size_t memory_source::size() const { return length; }
    
    void* memory_source::readmap(std::size_t pageoffset) const {
        byte* out = const_cast<byte*>(data);
        if (pageoffset) {
            out += pageoffset * ::getpagesize();
        }
        return static_cast<void*>(out);
    }
    
    memory_sink::memory_sink(byte* c, std::size_t len)
        :data(c)
        ,membuf(memory::sink(data, len))
        ,length(len)
        ,allocated(false)
        {}
    
    memory_sink::memory_sink(std::size_t len)
        :data(new byte[len])
        ,membuf(memory::sink(data, len))
        ,length(len)
        ,allocated(true)
        {}
    
    memory_sink::~memory_sink() {
        if (allocated) { delete[] data; }
    }
    
    bool memory_sink::can_seek() const noexcept { return true; }
    std::size_t memory_sink::seek_absolute(std::size_t pos) { std::fseek(membuf.get(), pos, SEEK_SET);
                                                              return std::ftell(membuf.get()); }
    
    std::size_t memory_sink::seek_relative(int delta) { std::fseek(membuf.get(), delta, SEEK_CUR);
                                                        return std::ftell(membuf.get()); }
    
    std::size_t memory_sink::seek_end(int delta) { std::fseek(membuf.get(), delta, SEEK_END);
                                                   return std::ftell(membuf.get()); }
    
    std::size_t memory_sink::write(const void* buffer, std::size_t n) {
        return std::fwrite(buffer, sizeof(byte), n, membuf.get());
    }
    
    // std::size_t memory_sink::write(std::vector<byte> const& bv) {
    //     return this->write(
    //         static_cast<const void*>(&bv[0]),
    //         bv.size());
    // }
    
    void memory_sink::flush() { std::fflush(membuf.get()); }
    
    std::vector<byte> memory_sink::contents() {
        std::vector<byte> out(length);
        std::fflush(membuf.get());
        std::memcpy(&out[0], data, length);
        return out;
    }
    
}
