/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ITERATORS_HH_
#define LIBIMREAD_ITERATORS_HH_

#include <utility>
#include <iterator>
#include <type_traits>

#include <libimread/libimread.hpp>

namespace im {
    
    class source_iterator {
        
        public:
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using idx_t = difference_type; /// Not into typing it out over and over
            using value_type = byte;
            using reference_type = std::add_lvalue_reference_t<value_type>;
            using pointer = std::add_pointer_t<value_type>;
            using iterator_category = std::random_access_iterator_tag;
            
            explicit source_iterator(byte_source* s)
                :source(s)
                ,sourcemap(static_cast<byte*>(source->readmap()))
                ,sourceidx(sourcemap)
                {}
            
            explicit source_iterator(byte_source* s, size_type initial_idx)
                :source(s)
                ,sourcemap(static_cast<byte*>(source->readmap()))
                ,sourceidx(sourcemap)
                {
                    sourcemap += initial_idx;
                }
            
            source_iterator(source_iterator const& other)
                :source(other.source)
                ,sourcemap(other.sourcemap)
                ,sourceidx(other.sourceidx)
                {}
            
            virtual ~source_iterator() {}
            
            source_iterator& operator=(source_iterator const& other) {
                source_iterator(other).swap(*this);
                return *this;
            }
            
            /// prefix increment
            source_iterator& operator++() {
                ++sourcemap;
                return *this;
            }
            
            /// postfix increment
            source_iterator operator++(int) {
                source_iterator out(*this);
                out.sourcemap++;
                return out;
            }
            
            /// prefix decrement
            source_iterator& operator--() {
                --sourcemap;
                return *this;
            }
            
            /// postfix decrement
            source_iterator operator--(int) {
                source_iterator out(*this);
                out.sourcemap--;
                return out;
            }
            
            source_iterator& operator+=(size_type offset) {
                sourcemap += offset;
                return *this;
            }
            
            source_iterator& operator-=(size_type offset) {
                sourcemap -= offset;
                return *this;
            }
            
            friend source_iterator operator+(source_iterator const& lhs, size_type rhs) {
                source_iterator out(lhs);
                out.sourcemap += rhs;
                return out;
            }
            
            friend source_iterator operator+(size_type lhs, source_iterator const& rhs) {
                source_iterator out(rhs);
                out.sourcemap += lhs;
                return out;
            }
            
            friend source_iterator operator-(source_iterator const& lhs, size_type rhs) {
                source_iterator out(lhs);
                out.sourcemap -= rhs;
                return out;
            }
            
            friend idx_t operator-(source_iterator lhs, source_iterator rhs) {
                return (idx_t)lhs.sourcemap - (idx_t)rhs.sourcemap;
            }
            
            value_type operator*() const {
                return sourcemap[0];
            }
            
            pointer operator->() const {
                return sourcemap;
            }
            
            reference_type operator[](size_type idx) const {
                return sourceidx[idx];
            }
            
            friend bool operator<(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap < rhs.sourcemap;
            }
            
            friend bool operator>(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap > rhs.sourcemap;
            }
            
            friend bool operator<=(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap <= rhs.sourcemap;
            }
            
            friend bool operator>=(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap >= rhs.sourcemap;
            }
            
            friend bool operator==(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap == rhs.sourcemap;
            }
            
            friend bool operator!=(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap != rhs.sourcemap;
            }
            
            void swap(source_iterator& other) {
                using std::swap;
                swap(source,    other.source);
                swap(sourcemap, other.sourcemap);
                swap(sourceidx, other.sourceidx);
            }
            
            friend void swap(source_iterator& lhs, source_iterator& rhs) {
                using std::swap;
                swap(lhs.source,    rhs.source);
                swap(lhs.sourcemap, rhs.sourcemap);
                swap(lhs.sourceidx, rhs.sourceidx);
            }
            
        private:
            byte_source* source;
            byte* sourcemap;
            byte* sourceidx;
    };
    
}


#endif /// LIBIMREAD_ITERATORS_HH_