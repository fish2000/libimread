/// Copyright 2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// URI encode and decode: RFC1630, RFC1738, RFC2396. Code adapted from Verizon’s HLX:
/// https://github.com/Verizon/hlx/blob/master/src/core/support/uri.cc#L1

#include <cstdint>
#include <libimread/ext/uri.hh>

namespace im {
    
    namespace uri {
        
        namespace detail {
            
            const int8_t HEX_TO_DECIMAL_LUT[256] = {
                /*       0  1  2  3   4  5  6  7   8  9  A  B   C  D  E  F */
                /* 0 */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* 1 */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* 2 */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* 3 */ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  -1, -1, -1, -1, -1, -1,
                
                /* 4 */ -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* 5 */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* 6 */ -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* 7 */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                
                /* 8 */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* 9 */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* A */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* B */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                
                /* C */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* D */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* E */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                /* F */ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
            };
            
            const char DECIMAL_TO_HEX_LUT[17] = "0123456789ABCDEF";
            
            /// Only alphanum is safe.
            const int8_t IS_SAFE_LUT[256] = {
                /*      0 1 2 3  4 5 6 7  8 9 A B  C D E F */
                /* 0 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* 1 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* 2 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* 3 */ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                
                /* 4 */ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                /* 5 */ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                /* 6 */ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                /* 7 */ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                
                /* 8 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* 9 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* A */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* B */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                
                /* C */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* D */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* E */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                /* F */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };
            
        } /// namespace detail
        
        std::string decode(std::string const& source) {
            /// N.B. from RFC1630:
            /// "Sequences which start with a percent sign but are not followed
            /// by two hexadecimal characters (0-9, A-F) are reserved for
            /// future extension"
            
            const unsigned char*       source_ptr    = reinterpret_cast<unsigned char const*>(source.c_str());
            const int                  length        = source.length();
            const unsigned char* const end           = source_ptr + length;
            const unsigned char* const lastdecodable = end - 2; /// last decodable '%'
            
            char* const start_ptr = new char[length];
            char*       end_ptr   = start_ptr;
            
            char dec1;
            char dec2;
            
            while (source_ptr < lastdecodable) {
                if (*source_ptr == '+') {
                    /// a space, encoded as a plus-sign:
                    *end_ptr = ' ';
                } else {
                    if (*source_ptr == '%') {
                        /// a hex-encoded char:
                        if ((-1 != (dec1 = static_cast<char>(detail::HEX_TO_DECIMAL_LUT[*(source_ptr + 1)]))) &&
                            (-1 != (dec2 = static_cast<char>(detail::HEX_TO_DECIMAL_LUT[*(source_ptr + 2)])))) {
                            *end_ptr = (dec1 << 4) + dec2;
                            ++end_ptr;
                            source_ptr += 3;
                            continue;
                        }
                    }
                    /// a non-hex-encoded char:
                    *end_ptr = *source_ptr;
                }
                ++end_ptr;
                ++source_ptr;
            }
            
            /// the remaining 2 chars:
            while (source_ptr < end) {
                *end_ptr = *source_ptr;
                ++end_ptr;
                ++source_ptr;
            }
            
            std::string out(start_ptr, end_ptr);
            delete[] start_ptr;
            return out;
        }
        
        std::string encode(const std::string& source) {
            const unsigned char*       source_ptr = reinterpret_cast<const unsigned char*>(source.c_str());
            const int                  length     = source.length();
            unsigned char* const       start_ptr  = new unsigned char[length * 3];
            unsigned char*             end_ptr    = start_ptr;
            const unsigned char* const end        = source_ptr + length;
            
            for (; source_ptr < end; ++source_ptr) {
                if (detail::IS_SAFE_LUT[*source_ptr]) {
                    /// safe, do not escape:
                    *end_ptr++ = *source_ptr;
                } else {
                    /// not safe, escape this char:
                    *end_ptr++ = '%';
                    *end_ptr++ = detail::DECIMAL_TO_HEX_LUT[*source_ptr >> 4];
                    *end_ptr++ = detail::DECIMAL_TO_HEX_LUT[*source_ptr & 0x0F];
                }
            }
            
            std::string out(reinterpret_cast<char*>(start_ptr),
                            reinterpret_cast<char*>(end_ptr));
            delete[] start_ptr;
            return out;
        }
        
    } /// namespace uri
    
} /// namespace im
