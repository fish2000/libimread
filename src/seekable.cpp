/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/seekable.hh>

namespace im {
    
    constexpr seekable::size_type seekable::kBufferSize;
    
    seekable::~seekable() {}
    
    bool seekable::can_seek() const noexcept { return false; }
    
    seekable::size_type seekable::seek_absolute(seekable::size_type) {
        imread_raise_default(NotImplementedError);
    }
    
    seekable::size_type seekable::seek_relative(int) {
        imread_raise_default(NotImplementedError);
    }
    
    seekable::size_type seekable::seek_end(int) {
        imread_raise_default(NotImplementedError);
    }
    
    seekable::size_type seekable::buffer_size() const noexcept {
        return kBufferSize;
    }
    
    byte_source::~byte_source() {}
    
    bytevec_t byte_source::full_data() const {
        bytevec_t result;
        byte_source::size_type n;
        byte buffer[kBufferSize];
        while ((n = read(buffer, sizeof(buffer)))) {
            result.insert(result.end(), buffer, buffer + n);
        }
        if (!__sized) {
            __siz = result.size();
            __sized = true;
        }
        return result;
    }
    
    byte_source::size_type byte_source::size() const {
        /// super-naive implementation...
        /// OVERRIDE THIS HORRIDNESS, DOGG
        if (!__sized) {
            bytevec_t all_of_it;
            byte_source::size_type n;
            byte buffer[kBufferSize];
            while ((n = read(buffer, sizeof(buffer)))) {
                all_of_it.insert(all_of_it.end(), buffer, buffer + n);
            }
            __siz = all_of_it.size();
            __sized = true;
        }
        return __siz;
    }
    
    bool byte_source::empty() const {
        return size() == 0;
    }
    
    byte* byte_source::data() const {
        return (byte*)readmap();
    }
    
    byte_source::iterator byte_source::begin() {
        return byte_source::iterator(data(), 0);
    }
    
    byte_source::iterator byte_source::end() {
        return byte_source::iterator(data(), size());
    }
    
    byte_source::const_iterator byte_source::begin() const {
        return byte_source::const_iterator(data(), 0);
    }
    
    byte_source::const_iterator byte_source::end() const {
        return byte_source::const_iterator(data(), size());
    }
    
    byte_source::reverse_iterator byte_source::rbegin() {
        return byte_source::reverse_iterator(byte_source::iterator(data(), size()));
    }
    
    byte_source::reverse_iterator byte_source::rend() {
        return byte_source::reverse_iterator(byte_source::iterator(data(), 0));
    }
    
    byte_source::const_reverse_iterator byte_source::rbegin() const {
        return byte_source::const_reverse_iterator(byte_source::const_iterator(data(), size()));
    }
    
    byte_source::const_reverse_iterator byte_source::rend() const {
        return byte_source::const_reverse_iterator(byte_source::const_iterator(data(), 0));
    }
    
    byte_sink::~byte_sink() {}
    
    byte_sink::size_type byte_sink::write(bytevec_t const& bytevec) {
        if (bytevec.empty()) { return 0; }
        return write(static_cast<const void*>(bytevec.data()), bytevec.size());
    }
    
    byte_sink::size_type byte_sink::write(bytevec_t&& bytevec) {
        if (bytevec.empty()) { return 0; }
        return write(static_cast<const void*>(bytevec.data()), bytevec.size());
    }
    
    void byte_sink::flush() {}

} /* namespace im */

namespace std {
    
    decltype(std::declval<im::byte_source>().begin())
        begin(im::byte_source* source) { return source->begin(); }
    
    decltype(std::declval<im::byte_source>().end())
        end(im::byte_source* source) { return source->end(); }
    
    decltype(std::declval<im::byte_source>().rbegin())
        rbegin(im::byte_source* source) { return source->rbegin(); }
    
    decltype(std::declval<im::byte_source>().rend())
        rend(im::byte_source* source) { return source->rend(); }
    
    decltype(std::declval<im::byte_source const>().begin())
        cbegin(im::byte_source const* source) { return source->begin(); }
    
    decltype(std::declval<im::byte_source const>().end())
        cend(im::byte_source const* source) { return source->end(); }
    
    decltype(std::declval<im::byte_source const>().rbegin())
        crbegin(im::byte_source const* source) { return source->rbegin(); }
    
    decltype(std::declval<im::byte_source const>().rend())
        crend(im::byte_source const* source) { return source->rend(); }
    
} /* namespace std */
