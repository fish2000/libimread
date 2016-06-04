/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ITERATORS_HH_
#define LIBIMREAD_ITERATORS_HH_

#include <utility>
#include <iterator>
#include <type_traits>

#include <libimread/libimread.hpp>

namespace im {
    
    /// forward-declare byte source and sink
    class byte_source;
    class byte_sink;
    
    class source_iterator {
        
        public:
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using idx_t = difference_type; /// Not into typing it out over and over
            using value_type = byte;
            using reference_type = std::add_lvalue_reference_t<value_type>;
            using pointer = std::add_pointer_t<value_type>;
            using iterator_category = std::random_access_iterator_tag;
            
            explicit source_iterator(byte_source* s);
            explicit source_iterator(byte_source* s, size_type initial_idx);
            source_iterator(source_iterator const& other);
            source_iterator(source_iterator&& other) noexcept;
            virtual ~source_iterator();
            
            source_iterator& operator=(source_iterator const& other);
            source_iterator& operator=(source_iterator&& other) noexcept;
            
            /// prefix increment
            source_iterator& operator++();
            
            /// postfix increment
            source_iterator operator++(int);
            
            /// prefix decrement
            source_iterator& operator--();
            
            /// postfix decrement
            source_iterator operator--(int);
            
            source_iterator& operator+=(size_type offset);
            source_iterator& operator-=(size_type offset);
            
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
            
            value_type operator*() const;
            pointer operator->() const;
            reference_type operator[](size_type idx) const;
            
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
            
            void swap(source_iterator& other);
            
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