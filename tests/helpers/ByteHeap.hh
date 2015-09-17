/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_TESTS_HELPERS_HEAPBYTES_HH_
#define LIBIMREAD_TESTS_HELPERS_HEAPBYTES_HH_

#include <cstdlib>
#include <string>
#include <vector>

#include <libimread/libimread.hpp>

namespace im {
namespace test {
    
    template <typename T = byte>
    class ByteHeap {
        
        public:
            using string_t = std::basic_string<T>;
            using vector_t = std::vector<T>;
            
            std::size_t siz;
            T* bytes;
            
            explicit ByteHeap(std::size_t s)
                :siz(s), bytes(new T[s])
                {
                    arc4random_buf(static_cast<void*>(bytes), siz);
                }
            
            ByteHeap(const ByteHeap& other)
                :siz(other.siz), bytes(other.bytes)
                {}
            
            virtual ~ByteHeap() {
                delete[] bytes;
            }
            
            std::size_t size() { return siz; }
            
            string_t bytestring() {
                return string_t(bytes);
            }
            
            vector_t bytevector() {
                vector_t out(siz);
                for (auto b : bytes) {
                    out.push_back(b);
                }
                return out;
            }
            
    };
    
    
}; /* namespace test */ 
}; /* namespace im */

#endif /// LIBIMREAD_TESTS_HELPERS_HEAPBYTES_HH_
