/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/iterators.hh>
#include <libimread/seekable.hh>

namespace im {
    
    source_iterator::source_iterator(byte_source* s)
        :source(s)
        ,sourcemap(static_cast<byte*>(source->readmap()))
        ,sourceidx(sourcemap)
        {}
    
    source_iterator::source_iterator(byte_source* s, size_type initial_idx)
        :source(s)
        ,sourcemap(static_cast<byte*>(source->readmap()))
        ,sourceidx(sourcemap)
        {
            sourcemap += initial_idx;
        }
    
    source_iterator::source_iterator(source_iterator const& other)
        :source(other.source)
        ,sourcemap(other.sourcemap)
        ,sourceidx(other.sourceidx)
        {}
    
    source_iterator::source_iterator(source_iterator&& other) noexcept
        :source(std::move(other.source))
        ,sourcemap(std::move(other.sourcemap))
        ,sourceidx(std::move(other.sourceidx))
        {}
    
    source_iterator::~source_iterator() {}
    
    source_iterator& source_iterator::operator=(source_iterator const& other) {
        source_iterator(other).swap(*this);
        return *this;
    }
    
    source_iterator& source_iterator::operator=(source_iterator&& other) noexcept {
        source = std::move(other.source);
        sourcemap = std::move(other.sourcemap);
        sourceidx = std::move(other.sourceidx);
        return *this;
    }
    
    /// prefix increment
    source_iterator& source_iterator::operator++() {
        ++sourcemap;
        return *this;
    }
    
    /// postfix increment
    source_iterator source_iterator::operator++(int) {
        source_iterator out(*this);
        out.sourcemap++;
        return out;
    }
    
    /// prefix decrement
    source_iterator& source_iterator::operator--() {
        --sourcemap;
        return *this;
    }
    
    /// postfix decrement
    source_iterator source_iterator::operator--(int) {
        source_iterator out(*this);
        out.sourcemap--;
        return out;
    }
    
    source_iterator& source_iterator::operator+=(size_type offset) {
        sourcemap += offset;
        return *this;
    }
    
    source_iterator& source_iterator::operator-=(size_type offset) {
        sourcemap -= offset;
        return *this;
    }
    
    // friend source_iterator operator+(source_iterator const& lhs, size_type rhs) {
    //     source_iterator out(lhs);
    //     out.sourcemap += rhs;
    //     return out;
    // }
    //
    // friend source_iterator operator+(size_type lhs, source_iterator const& rhs) {
    //     source_iterator out(rhs);
    //     out.sourcemap += lhs;
    //     return out;
    // }
    //
    // friend source_iterator operator-(source_iterator const& lhs, size_type rhs) {
    //     source_iterator out(lhs);
    //     out.sourcemap -= rhs;
    //     return out;
    // }
    //
    // friend idx_t operator-(source_iterator lhs, source_iterator rhs) {
    //     return (idx_t)lhs.sourcemap - (idx_t)rhs.sourcemap;
    // }
    
    source_iterator::value_type source_iterator::operator*() const {
        return sourcemap[0];
    }
    
    source_iterator::pointer source_iterator::operator->() const {
        return sourcemap;
    }
    
    source_iterator::reference_type source_iterator::operator[](size_type idx) const {
        return sourceidx[idx];
    }
    
    // friend bool operator<(source_iterator const& lhs, source_iterator const& rhs) {
    //     return lhs.sourcemap < rhs.sourcemap;
    // }
    //
    // friend bool operator>(source_iterator const& lhs, source_iterator const& rhs) {
    //     return lhs.sourcemap > rhs.sourcemap;
    // }
    //
    // friend bool operator<=(source_iterator const& lhs, source_iterator const& rhs) {
    //     return lhs.sourcemap <= rhs.sourcemap;
    // }
    //
    // friend bool operator>=(source_iterator const& lhs, source_iterator const& rhs) {
    //     return lhs.sourcemap >= rhs.sourcemap;
    // }
    //
    // friend bool operator==(source_iterator const& lhs, source_iterator const& rhs) {
    //     return lhs.sourcemap == rhs.sourcemap;
    // }
    //
    // friend bool operator!=(source_iterator const& lhs, source_iterator const& rhs) {
    //     return lhs.sourcemap != rhs.sourcemap;
    // }
    
    void source_iterator::swap(source_iterator& other) {
        using std::swap;
        swap(source,    other.source);
        swap(sourcemap, other.sourcemap);
        swap(sourceidx, other.sourceidx);
    }
    
    // friend void swap(source_iterator& lhs, source_iterator& rhs) {
    //     using std::swap;
    //     swap(lhs.source,    rhs.source);
    //     swap(lhs.sourcemap, rhs.sourcemap);
    //     swap(lhs.sourceidx, rhs.sourceidx);
    // }
    
}
