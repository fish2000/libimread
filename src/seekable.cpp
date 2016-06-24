/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/seekable.hh>
#include <libimread/iterators.hh>

namespace im {
    
    seekable::~seekable() {}
    
    bool seekable::can_seek() const noexcept { return false; }
    
    std::size_t seekable::seek_absolute(std::size_t) {
        imread_raise_default(NotImplementedError);
    }
    
    std::size_t seekable::seek_relative(int) {
        imread_raise_default(NotImplementedError);
    }
    
    std::size_t seekable::seek_end(int) {
        imread_raise_default(NotImplementedError);
    }
    
    byte_source::~byte_source() {}
    
    std::vector<byte> byte_source::full_data() {
        std::vector<byte> result;
        std::size_t n;
        byte buffer[4096];
        while ((n = this->read(buffer, sizeof(buffer)))) {
            result.insert(result.end(), buffer, buffer + n);
        }
        return result;
    }
    
    std::size_t byte_source::size() const {
        /// super-naive implementation...
        /// OVERRIDE THIS HORRIDNESS, DOGG
        std::vector<byte> all_of_it;
        std::size_t n;
        byte buffer[4096];
        byte_source* mutablethis = const_cast<byte_source*>(this);
        while ((n = mutablethis->read(buffer, sizeof(buffer)))) {
            all_of_it.insert(all_of_it.end(), buffer, buffer + n);
        }
        return all_of_it.size();
    }
    
    bool byte_source::empty() const {
        return size() == 0;
    }
    
    byte* byte_source::data() const {
        return (byte*)readmap();
    }
    
    byte_source::iterator byte_source::begin() {
        return byte_source::iterator(this, 0);
    }
    
    byte_source::iterator byte_source::end() {
        return byte_source::iterator(this, size());
    }
    
    byte_source::const_iterator byte_source::begin() const {
        return byte_source::const_iterator(this, 0);
    }
    
    byte_source::const_iterator byte_source::end() const {
        return byte_source::const_iterator(this, size());
    }
    
    byte_source::reverse_iterator byte_source::rbegin() {
        return byte_source::reverse_iterator(byte_source::iterator(this, size()));
    }
    
    byte_source::reverse_iterator byte_source::rend() {
        return byte_source::reverse_iterator(byte_source::iterator(this, 0));
    }
    
    byte_source::const_reverse_iterator byte_source::rbegin() const {
        return byte_source::const_reverse_iterator(byte_source::const_iterator(this, size()));
    }
    
    byte_source::const_reverse_iterator byte_source::rend() const {
        return byte_source::const_reverse_iterator(byte_source::const_iterator(this, 0));
    }
    
    byte_sink::~byte_sink() {}
    void byte_sink::flush() {}

} /* namespace im */

namespace std {
    
    using im::byte_source;
    
    decltype(std::declval<byte_source>().begin())
        begin(byte_source* source) { return source->begin(); }
    
    decltype(std::declval<byte_source>().end())
        end(byte_source* source) { return source->end(); }
    
    decltype(std::declval<byte_source>().rbegin())
        rbegin(byte_source* source) { return source->rbegin(); }
    
    decltype(std::declval<byte_source>().rend())
        rend(byte_source* source) { return source->rend(); }
    
    decltype(std::declval<byte_source const>().begin())
        cbegin(byte_source const* source) { return source->begin(); }
    
    decltype(std::declval<byte_source const>().end())
        cend(byte_source const* source) { return source->end(); }
    
    decltype(std::declval<byte_source const>().rbegin())
        crbegin(byte_source const* source) { return source->rbegin(); }
    
    decltype(std::declval<byte_source const>().rend())
        crend(byte_source const* source) { return source->rend(); }
    
} /* namespace std */
