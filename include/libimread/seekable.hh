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
        
        public:
            using value_type = byte;
            using difference_type = std::ptrdiff_t;
            using size_type = std::size_t;
        
        protected:
            static constexpr size_type kBufferSize = 4096;
        
        public:
            using reference_type = std::add_lvalue_reference_t<value_type>;
            using reference = std::add_lvalue_reference_t<value_type>;
            using const_reference = std::add_const_t<reference>;
            using pointer = std::add_pointer_t<value_type>;
            using vector_type = std::vector<value_type>;
        
        public:
            using iterator = byte_iterator;
            using const_iterator = byte_iterator;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        
        public:
            virtual ~seekable();
            virtual bool can_seek() const noexcept;
            virtual size_type seek_absolute(size_type);
            virtual size_type seek_relative(int);
            virtual size_type seek_end(int);
        
        public:
            virtual size_type buffer_size() const noexcept;
        
    };
    
    class byte_source : virtual public seekable {
        
        protected:
            using seekable::kBufferSize;
        
        public:
            using seekable::value_type;
            using seekable::difference_type;
            using seekable::size_type;
            using seekable::reference_type;
            using seekable::reference;
            using seekable::const_reference;
            using seekable::pointer;
            using seekable::vector_type;
            using seekable::iterator;
            using seekable::const_iterator;
            using seekable::reverse_iterator;
            using seekable::const_reverse_iterator;
        
        public:
            virtual ~byte_source();
            virtual size_type read(pointer, size_type) const warn_unused = 0;
            virtual void* readmap(size_type pageoffset = 0) const = 0;
            virtual vector_type full_data() const;
            virtual size_type size() const;
            bool empty() const;
            pointer data() const;
            
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
        
        protected:
            using seekable::kBufferSize;
        
        public:
            using seekable::value_type;
            using seekable::difference_type;
            using seekable::size_type;
            using seekable::reference_type;
            using seekable::reference;
            using seekable::const_reference;
            using seekable::pointer;
            using seekable::vector_type;
            using seekable::iterator;
            using seekable::const_iterator;
            using seekable::reverse_iterator;
            using seekable::const_reverse_iterator;
        
        public:
            virtual ~byte_sink();
            virtual size_type write(const void*, size_type) = 0;
            virtual size_type write(vector_type const&);
            virtual size_type write(vector_type&&);
            virtual void flush();
        
        public:
            void push_back(value_type const&);
            void push_back(value_type&&);
            
        public:
            void push_front(value_type const&);
            void push_front(value_type&&);
        
        public:
            template <typename ...Args>
            __attribute__((nonnull(2)))
            size_type writef(char const* format, Args... args) {
                char buffer[kBufferSize];
                std::snprintf(buffer, sizeof(buffer), format, args...);
                return write(buffer, std::strlen(
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