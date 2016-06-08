/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// This is Luis Pedro's implementation of TIFF LZW compression, coded from the spec.

#ifndef LIBIMREAD_EXT_LZW_HH_
#define LIBIMREAD_EXT_LZW_HH_

#include <vector>
#include <string>
#include <libimread/libimread.hpp>

namespace im {
    
    namespace lzw {
        
        using stringvec_t = std::vector<std::string>;
        using bytevec_t = std::vector<byte>;
        
        struct code_stream {
            
            public:
                code_stream(byte* buf, unsigned long len);
                unsigned short getbit();
                unsigned short get(int nbits);
            
            private:
                const byte* buf_;
                int byte_pos_;
                int bit_pos_;
                const int len_;
            
        };
        
        namespace detail {
            std::string table_at(stringvec_t const& table, unsigned short idx);
            void write_string(bytevec_t& output, std::string const& s);
        }
        
        /// this is the main public-facing API function
        bytevec_t decode(void* buf, unsigned long len);
        
    } /* namespace lzw */
    
} /* namespace im */

#endif /// LIBIMREAD_EXT_LZW_HH_