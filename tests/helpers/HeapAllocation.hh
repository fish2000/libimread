/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_TESTS_HELPERS_HEAPALLOCATION_HH_
#define LIBIMREAD_TESTS_HELPERS_HEAPALLOCATION_HH_
#define CATCH_CONFIG_FAST_COMPILE

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <libimread/libimread.hpp>

namespace im {
namespace test {
    
    template <typename T>
    class HeapAllocation {
        
        public:
            using string_t = std::basic_string<T>;
            using vector_t = std::vector<T>;
            
            std::size_t len;
            std::size_t siz;
            bool deallocate;
            T* data;
            
            explicit HeapAllocation(std::size_t s)
                :len(s)
                ,siz(s * sizeof(T))
                ,deallocate(true)
                ,data(new T[len])
                {
                    arc4random_buf(static_cast<void*>(data), siz);
                }
            
            HeapAllocation(HeapAllocation const& other)
                :len(other.len)
                ,siz(other.siz)
                ,deallocate(true)
                ,data(new T[len])
                {
                    std::memcpy(other.data, data, siz);
                }
            
            HeapAllocation(HeapAllocation&& other) noexcept
                :len(other.len)
                ,siz(other.siz)
                ,deallocate(true)
                ,data(std::move(other.data))
                {
                    other.deallocate = false;
                }
            
            virtual ~HeapAllocation() {
                if (deallocate) {
                    delete[] data;
                }
            }
            
            std::size_t size() const noexcept { return siz; }
            std::size_t length() const noexcept { return len; }
            
            string_t typedstring() const { return string_t(data, siz); }
            std::string string() const { return std::string((char const*)data, siz); }
            
            vector_t vector() const {
                vector_t out;
                out.reserve(len);
                for (int idx = 0; idx < len; ++idx) {
                    out.emplace_back(data[idx]);
                }
                return out;
            }
            
    };
    
    using Bytes = HeapAllocation<byte>;
    using Chars = HeapAllocation<char>;
    
    
}; /* namespace test */ 
}; /* namespace im */

#endif /// LIBIMREAD_TESTS_HELPERS_HEAPALLOCATION_HH_