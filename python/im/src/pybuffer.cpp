/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <cstring>
#include "pybuffer.hpp"

namespace py {
    
    namespace buffer {
            
            source::source(Py_buffer const& pb)
                :view(pb), pos(0), release(true)
                {}
            
            source::source(Py_buffer const& pb, bool r)
                :view(pb), pos(0), release(r)
                {}
            
            source::~source() {
                if (release) {
                    PyBuffer_Release(&view);
                }
            }
            
            std::size_t source::read(byte* buffer, std::size_t n) {
                if (pos + n > view.len) { n = view.len - pos; }
                std::memmove(buffer, (byte*)view.buf + pos, n);
                pos += n;
                return n;
            }
            
            bool source::can_seek() const noexcept { return true; }
            
            std::size_t source::seek_absolute(std::size_t p) {
                return pos = p; 
            }
            std::size_t source::seek_relative(int delta) {
                return pos += delta;
            }
            std::size_t source::seek_end(int delta) {
                return pos = (view.len-delta-1);
            }
            
            std::vector<byte> source::full_data() {
                std::vector<byte> out(view.len);
                std::memcpy(&out[0], view.buf, view.len);
                return out;
            }
            
            std::size_t source::size() const {
                return view.len;
            }
            
            void* source::readmap(std::size_t pageoffset) const {
                byte* out = (byte*)view.buf;
                if (pageoffset) {
                    out += pageoffset * ::getpagesize();
                }
                return static_cast<void*>(out);
            }
            
            std::string source::str() const {
                return std::string((char const*)view.buf, view.len);
            }
            
            
            sink::sink(Py_buffer& pb)
                :view(pb), pos(0), release(false)
                {}
            
            sink::sink(Py_buffer& pb, bool r)
                :view(pb), pos(0), release(r)
                {}
            
            sink::~sink() {
                if (release) {
                    PyBuffer_Release(&view);
                }
            }
            
            bool sink::can_seek() const noexcept { return true; }
            
            std::size_t sink::seek_absolute(std::size_t p) {
                return pos = p;
            }
            std::size_t sink::seek_relative(int delta) {
                return pos += delta;
            }
            std::size_t sink::seek_end(int delta) {
                return pos = (view.len-delta-1);
            }
            
            std::size_t sink::write(const void* buffer, std::size_t n) {
                std::memmove((byte*)view.buf + pos, (byte*)buffer, n);
                pos += n;
                return n;
            }
            
            void sink::flush() {}
            
            std::vector<byte> sink::contents() const {
                std::vector<byte> out(view.len);
                std::memcpy(&out[0], view.buf, view.len);
                return out;
            }
            
            std::string sink::str() const {
                return std::string((char const*)view.buf, view.len);
            }
            
    }
    
}

