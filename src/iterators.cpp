/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/iterators.hh>

namespace im {
    
    byte_iterator::byte_iterator(void)
        :sourcemap{ 0 }
        {}
    
    byte_iterator::byte_iterator(byte_iterator::pointer byteptr)
        :sourcemap{ byteptr }
        {}
    
    byte_iterator::byte_iterator(byte_iterator::pointer byteptr, size_type initial_idx)
        :sourcemap{ byteptr + initial_idx }
        {}
    
    byte_iterator::byte_iterator(byte_iterator const& other)
        :sourcemap{ other.sourcemap }
        {}
    
    byte_iterator::byte_iterator(byte_iterator&& other) noexcept
        :sourcemap{ other.sourcemap }
        {}
    
    byte_iterator::~byte_iterator() {}
    
    byte_iterator& byte_iterator::operator=(byte_iterator const& other) {
        if (sourcemap != other.sourcemap) {
            byte_iterator(other).swap(*this);
        }
        return *this;
    }
    
    byte_iterator& byte_iterator::operator=(byte_iterator&& other) noexcept {
        if (sourcemap != other.sourcemap) {
            byte_iterator(other).swap(*this);
        }
        return *this;
    }
    
    byte_iterator& byte_iterator::operator,(byte_iterator const& other) {
        *this += other;
        return *this;
    }
    
    byte_iterator& byte_iterator::operator,(byte_iterator&& other) {
        *this += other;
        return *this;
    }
    
    /// prefix increment
    byte_iterator& byte_iterator::operator++() {
        ++sourcemap;
        return *this;
    }
    
    /// postfix increment
    byte_iterator byte_iterator::operator++(int) {
        byte_iterator out(*this);
        ++sourcemap;
        return out;
    }
    
    /// prefix decrement
    byte_iterator& byte_iterator::operator--() {
        --sourcemap;
        return *this;
    }
    
    /// postfix decrement
    byte_iterator byte_iterator::operator--(int) {
        byte_iterator out(*this);
        --sourcemap;
        return out;
    }
    
    byte_iterator& byte_iterator::operator+=(byte_iterator const& offset) {
        sourcemap += reinterpret_cast<byte_iterator::size_type>(offset.sourcemap);
        return *this;
    }
    
    byte_iterator& byte_iterator::operator-=(byte_iterator const& offset) {
        sourcemap -= reinterpret_cast<byte_iterator::size_type>(offset.sourcemap);
        return *this;
    }
    
    byte_iterator& byte_iterator::operator+=(size_type offset) {
        sourcemap += offset;
        return *this;
    }
    
    byte_iterator& byte_iterator::operator-=(size_type offset) {
        sourcemap -= offset;
        return *this;
    }
    
    byte_iterator operator+(byte_iterator const& lhs, byte_iterator::size_type rhs) {
        byte_iterator out(lhs);
        out.sourcemap += rhs;
        return out;
    }
    
    byte_iterator operator+(byte_iterator::size_type lhs, byte_iterator const& rhs) {
        byte_iterator out(rhs);
        out.sourcemap += lhs;
        return out;
    }
    
    byte_iterator operator-(byte_iterator const& lhs, byte_iterator::size_type rhs) {
        byte_iterator out(lhs);
        out.sourcemap -= rhs;
        return out;
    }
    
    byte_iterator::idx_t operator-(byte_iterator lhs, byte_iterator rhs) {
        return (byte_iterator::idx_t)lhs.sourcemap - (byte_iterator::idx_t)rhs.sourcemap;
    }
    
    byte_iterator::reference_type byte_iterator::operator*() const {
        return *sourcemap;
    }
    
    byte_iterator::pointer byte_iterator::operator->() const {
        return sourcemap;
    }
    
    byte_iterator::pointer byte_iterator::operator&() const {
        return sourcemap;
    }
    
    byte_iterator::reference_type byte_iterator::operator[](size_type idx) const {
        return sourcemap[idx];
    }
    
    bool operator<(byte_iterator const& lhs, byte_iterator const& rhs) {
        return lhs.sourcemap < rhs.sourcemap;
    }
    
    bool operator>(byte_iterator const& lhs, byte_iterator const& rhs) {
        return lhs.sourcemap > rhs.sourcemap;
    }
    
    bool operator<=(byte_iterator const& lhs, byte_iterator const& rhs) {
        return lhs.sourcemap <= rhs.sourcemap;
    }
    
    bool operator>=(byte_iterator const& lhs, byte_iterator const& rhs) {
        return lhs.sourcemap >= rhs.sourcemap;
    }
    
    bool operator==(byte_iterator const& lhs, byte_iterator const& rhs) {
        return lhs.sourcemap == rhs.sourcemap;
    }
    
    bool operator!=(byte_iterator const& lhs, byte_iterator const& rhs) {
        return lhs.sourcemap != rhs.sourcemap;
    }
    
    void byte_iterator::swap(byte_iterator& other) {
        using std::swap;
        swap(sourcemap, other.sourcemap);
    }
    
    void swap(byte_iterator& lhs, byte_iterator& rhs) {
        using std::swap;
        swap(lhs.sourcemap, rhs.sourcemap);
    }
    
} /* namespace im */

