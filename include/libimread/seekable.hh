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
    
    /// forward declarations
    class source_iterator;
    class source_const_iterator;
    
    struct seekable {
        virtual ~seekable();
        virtual bool can_seek() const noexcept;
        virtual std::size_t seek_absolute(std::size_t);
        virtual std::size_t seek_relative(int);
        virtual std::size_t seek_end(int);
    };
    
    class byte_source : virtual public seekable {
        
        public:
            using value_type = byte;
            
            using difference_type = std::ptrdiff_t;
            using size_type = std::size_t;
            using reference_type = std::add_lvalue_reference_t<value_type>;
            using reference = std::add_lvalue_reference_t<value_type>;
            using const_reference = std::add_const_t<reference>;
            using pointer = std::add_pointer_t<value_type>;
            
            using iterator = source_iterator;
            using const_iterator = source_iterator;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            
            virtual ~byte_source();
            virtual std::size_t read(byte* buffer, std::size_t) warn_unused = 0;
            virtual void* readmap(std::size_t pageoffset = 0) const = 0;
            
            // void read_check(byte* buffer, std::size_t n) {
            //     if (this->read(buffer, n) != n) {
            //         imread_raise(CannotReadError,
            //             "File ended prematurely");
            //     }
            // }
            
            virtual std::vector<byte> full_data();
            virtual std::size_t size() const;
            bool empty() const;
            byte* data() const;
            
            iterator begin();
            iterator end();
            const_iterator begin() const;
            const_iterator end() const;
            reverse_iterator rbegin();
            reverse_iterator rend();
            const_reverse_iterator rbegin() const;
            const_reverse_iterator rend() const;
    };
    
    class byte_sink : virtual public seekable {
        
        public:
            using value_type = byte;
            using difference_type = std::ptrdiff_t;
            using size_type = std::size_t;
            using reference = std::add_lvalue_reference_t<value_type>;
            using const_reference = std::add_const_t<reference>;
            // using iterator = source_iterator;
            // using const_iterator = source_iterator;
            // using reverse_iterator = std::reverse_iterator<iterator>;
            // using const_reverse_iterator = std::reverse_iterator<const_iterator>;
            
            virtual ~byte_sink();
            virtual std::size_t write(const void* buffer, std::size_t n) = 0;
            
            // void write_check(const byte* buffer, std::size_t n) {
            //     std::size_t out = this->write(buffer, n);
            //     imread_assert(out == n,
            //         "write_check() return value differs from n:",
            //             FF("\t  n = %i", n),
            //             FF("\tout = %i", out));
            // }
            
            virtual void flush();
            
            template <typename ...Args>
            std::size_t writef(const char* format, Args... args) {
                char buffer[1024];
                std::snprintf(buffer, 1024, format, args...);
                return this->write(buffer, std::strlen(
                                   static_cast<const char*>(buffer)));
            }
    };

} /* namespace im */

namespace std {
    
    using im::byte_source;
    
    decltype(std::declval<byte_source>().begin())
        begin(byte_source*);
    
    decltype(std::declval<byte_source>().end())
        end(byte_source*);
    
    decltype(std::declval<byte_source>().rbegin())
        rbegin(byte_source*);
    
    decltype(std::declval<byte_source>().rend())
        rend(byte_source*);
    
    decltype(std::declval<byte_source const>().begin())
        cbegin(byte_source const*);
    
    decltype(std::declval<byte_source const>().end())
        cend(byte_source const*);
    
    decltype(std::declval<byte_source const>().rbegin())
        crbegin(byte_source const*);
    
    decltype(std::declval<byte_source const>().rend())
        crend(byte_source const*);
    
} /* namespace std */


#endif /// LIBIMREAD_SEEKABLE_HH_