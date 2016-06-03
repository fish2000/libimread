/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ITERATORS_HH_
#define LIBIMREAD_ITERATORS_HH_

#include <stack>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/seekable.hh>

namespace im {
    
    class source_iterator {
        
        public:
            using size_type = std::size_t
            using difference_type = std::ptrdiff_t;
            using idx_t = difference_type; /// Not into typing it out over and over
            using value_type = byte;
            using reference = std::add_lvalue_reference_t<value_type>;
            using pointer = std::add_pointer_t<value_type>;
            using iterator_category = std::random_access_iterator_tag;
            
            explicit source_iterator(byte_source* s)
                :source(s)
                ,sourcemap(source->readmap())
                {}
            
            source_iterator(source_iterator const& other)
                :source(other.source)
                ,sourcemap(other.sourcemap)
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
            
            
            
            value_type operator*() const {
                return *sourcemap;
            }
            
            pointer operator->() const {
                return sourcemap;
            }
            
            friend bool operator==(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap == rhs.sourcemap;
            }
            
            friend bool operator!=(source_iterator const& lhs, source_iterator const& rhs) {
                return lhs.sourcemap != rhs.sourcemap;
            }
            
            friend void swap(source_iterator& lhs, source_iterator& rhs) {
                using std::swap;
                swap(lhs.source,    rhs.source);
                swap(lhs.sourcemap, rhs.sourcemap);
            }
            
        private:
            byte_source* source;
            byte* sourcemap;
    };
    
}


#endif /// LIBIMREAD_ITERATORS_HH_