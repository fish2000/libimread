/// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_BASE64_HH_
#define LIBIMREAD_EXT_BASE64_HH_

#include <string>
#include <vector>
#include <memory>
#include <libimread/libimread.hpp>

namespace im {
    
    namespace base64 {
        
        using bytevec_t = std::vector<byte>;
        using charptr_t = std::unique_ptr<char[]>;
        
        void encode(std::string& out, bytevec_t const& buf);
        void encode(std::string& out, byte const* buf, std::size_t bufLen);
        void encode(std::string& out, std::string const& buf);
        
        std::string encode(byte const* buf, std::size_t bufLen);
        std::string encode(char const* cstring, std::size_t bufLen);
        std::string encode(char const* cstring);
        
        /// use this if you know the output should be a valid string
        void decode(std::string& out, std::string const& encoded_string);
        void decode(bytevec_t& out, std::string const& encoded_string);
        
        charptr_t decode(std::string const& encoded_string);
        
    }
    
}

#endif /// LIBIMREAD_EXT_BASE64_HH_