/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_SEEKABLE_HH_
#define LIBIMREAD_SEEKABLE_HH_

#include <cstdio>
#include <vector>
#include <utility>
#include <iterator>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/iterators.hh>

namespace im {
    
    struct seekable {
        
        using value_type = byte;
        using difference_type = std::ptrdiff_t;
        using size_type = std::size_t;
        
        using reference_type = std::add_lvalue_reference_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using const_reference = std::add_const_t<reference>;
        using pointer = std::add_pointer_t<value_type>;
        
        using iterator = byte_iterator;
        using const_iterator = byte_iterator;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        
        virtual ~seekable();
        virtual bool can_seek() const noexcept;
        virtual std::size_t seek_absolute(std::size_t);
        virtual std::size_t seek_relative(int);
        virtual std::size_t seek_end(int);
        
    };
    
    class byte_source : virtual public seekable {
        
        public:
            using seekable::value_type;
            using seekable::difference_type;
            using seekable::size_type;
            using seekable::reference_type;
            using seekable::reference;
            using seekable::const_reference;
            using seekable::pointer;
            using seekable::iterator;
            using seekable::const_iterator;
            using seekable::reverse_iterator;
            using seekable::const_reverse_iterator;
        
        public:
            virtual ~byte_source();
            virtual std::size_t read(byte*, std::size_t) warn_unused = 0;
            virtual void* readmap(std::size_t pageoffset = 0) const = 0;
            virtual bytevec_t full_data();
            virtual std::size_t size() const;
            bool empty() const;
            byte* data() const;
            
        public:
            iterator begin();
            iterator end();
            const_iterator begin() const;
            const_iterator end() const;
            reverse_iterator rbegin();
            reverse_iterator rend();
            const_reverse_iterator rbegin() const;
            const_reverse_iterator rend() const;
        
        private:
            mutable bool __sized = false;
            mutable size_type __siz = 0;
    };
    
    class byte_sink : virtual public seekable {
        
        public:
            using seekable::value_type;
            using seekable::difference_type;
            using seekable::size_type;
            using seekable::reference_type;
            using seekable::reference;
            using seekable::const_reference;
            using seekable::pointer;
            using seekable::iterator;
            using seekable::const_iterator;
            using seekable::reverse_iterator;
            using seekable::const_reverse_iterator;
            
        public:
            virtual ~byte_sink();
            virtual std::size_t write(const void*, std::size_t) = 0;
            virtual std::size_t write(bytevec_t const&);
            virtual void flush();
            
        public:
            template <typename ...Args>
            __attribute__((nonnull (2)))
            std::size_t writef(char const* format, Args... args) {
                char buffer[1024];
                std::snprintf(buffer, 1024, format, args...);
                return this->write(buffer, std::strlen(
                              static_cast<char const*>(buffer)));
            }
    };

} /* namespace im */

namespace std {
    
    decltype(std::declval<im::byte_source>().begin())
        begin(im::byte_source*);
    
    decltype(std::declval<im::byte_source>().end())
        end(im::byte_source*);
    
    decltype(std::declval<im::byte_source>().rbegin())
        rbegin(im::byte_source*);
    
    decltype(std::declval<im::byte_source>().rend())
        rend(im::byte_source*);
    
    decltype(std::declval<im::byte_source const>().begin())
        cbegin(im::byte_source const*);
    
    decltype(std::declval<im::byte_source const>().end())
        cend(im::byte_source const*);
    
    decltype(std::declval<im::byte_source const>().rbegin())
        crbegin(im::byte_source const*);
    
    decltype(std::declval<im::byte_source const>().rend())
        crend(im::byte_source const*);
    
} /* namespace std */

#endif /// LIBIMREAD_SEEKABLE_HH_