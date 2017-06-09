/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ITERATORS_HH_
#define LIBIMREAD_ITERATORS_HH_

#include <utility>
#include <iterator>
#include <type_traits>

#include <libimread/libimread.hpp>

namespace im {
    
    class byte_iterator {
        
        public:
            using value_type = byte;
            using iterator_category = std::random_access_iterator_tag;
            
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using idx_t = difference_type; /// Not into typing it out over and over
            using reference_type = std::add_lvalue_reference_t<value_type>;
            using reference = std::add_lvalue_reference_t<value_type>;
            using const_reference = std::add_const_t<reference>;
            using pointer = std::add_pointer_t<value_type>;
            
            explicit byte_iterator(pointer byteptr);
            explicit byte_iterator(pointer byteptr, size_type initial_idx);
            byte_iterator(byte_iterator const& other);
            byte_iterator(byte_iterator&& other) noexcept;
            virtual ~byte_iterator();
            
            byte_iterator& operator=(byte_iterator const& other);
            byte_iterator& operator=(byte_iterator&& other) noexcept;
            
            /// prefix increment
            byte_iterator& operator++();
            
            /// postfix increment
            byte_iterator operator++(int);
            
            /// prefix decrement
            byte_iterator& operator--();
            
            /// postfix decrement
            byte_iterator operator--(int);
            
            byte_iterator& operator+=(size_type offset);
            byte_iterator& operator-=(size_type offset);
            
            friend byte_iterator operator+(byte_iterator const& lhs, size_type rhs);
            friend byte_iterator operator+(size_type lhs, byte_iterator const& rhs);
            friend byte_iterator operator-(byte_iterator const& lhs, size_type rhs);
            friend idx_t operator-(byte_iterator lhs, byte_iterator rhs);
            
            reference_type operator*() const;
            pointer operator->() const;
            pointer operator&() const;
            reference_type operator[](size_type idx) const;
            
            friend bool operator<(byte_iterator const& lhs, byte_iterator const& rhs);
            friend bool operator>(byte_iterator const& lhs, byte_iterator const& rhs);
            friend bool operator<=(byte_iterator const& lhs, byte_iterator const& rhs);
            friend bool operator>=(byte_iterator const& lhs, byte_iterator const& rhs);
            friend bool operator==(byte_iterator const& lhs, byte_iterator const& rhs);
            friend bool operator!=(byte_iterator const& lhs, byte_iterator const& rhs);
            
            void swap(byte_iterator& other);
            friend void swap(byte_iterator& lhs, byte_iterator& rhs);
            
        protected:
            pointer sourcemap;
    };
    
} /* namespace im */


#endif /// LIBIMREAD_ITERATORS_HH_