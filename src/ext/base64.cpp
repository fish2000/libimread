
#include <cstring>
#include <libimread/ext/base64.hh>

namespace im {
    
    namespace base64 {
        
        /// implementation thanks to elegant dice of SO:
        /// http://stackoverflow.com/a/35328409/298171
        
        static const byte from_base64[128] = {
            /// 8 rows of 16 = 128
            /// note: only require 123 entries, as we only lookup for <= z , which z=122
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  62, 255,  62, 255,  63,
            52,   53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255,   0, 255, 255, 255,
            255,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
            15,   16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255,  63,
            255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
            41,   42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 255, 255, 255, 255, 255
        };
        
        static const char to_base64[65] = 
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "0123456789+/";
        
        void encode(std::string& out, std::string const& buf) {
            if (buf.empty()) {
                encode(out, nullptr, 0);
            } else {
                encode(out, reinterpret_cast<byte const*>(&buf[0]), buf.size());
            }
        }
        
        void encode(std::string& out, bytevec_t const& buf) {
            if (buf.empty()) {
                encode(out, nullptr, 0);
            } else {
                encode(out, &buf[0], buf.size());
            }
        }
        
        void encode(std::string& out, byte const* buf, std::size_t bufLen) {
            // Calculate how many bytes that needs to be added to get a multiple of 3
            std::size_t missing = 0;
            std::size_t ret_size = bufLen;
            while ((ret_size % 3) != 0) {
                ++ret_size;
                ++missing;
            }
            
            /// Expand the return string size to a multiple of 4
            ret_size = 4 * ret_size / 3;
            out.clear();
            out.reserve(ret_size);
            
            for (std::size_t idx = 0; idx < ret_size/4; ++idx) {
                /// Read a group of three bytes (avoid buffer overrun by replacing with 0)
                const std::size_t buffer_index = idx * 3;
                const byte b3_0 = (buffer_index+0 < bufLen) ? buf[buffer_index+0] : 0;
                const byte b3_1 = (buffer_index+1 < bufLen) ? buf[buffer_index+1] : 0;
                const byte b3_2 = (buffer_index+2 < bufLen) ? buf[buffer_index+2] : 0;
                
                /// Transform into four base 64 characters
                const byte b4_0 =                        ((b3_0 & 0xfc) >> 2);
                const byte b4_1 = ((b3_0 & 0x03) << 4) + ((b3_1 & 0xf0) >> 4);
                const byte b4_2 = ((b3_1 & 0x0f) << 2) + ((b3_2 & 0xc0) >> 6);
                const byte b4_3 = ((b3_2 & 0x3f) << 0);
                
                /// Add the base 64 characters to the return value
                out.push_back(to_base64[b4_0]);
                out.push_back(to_base64[b4_1]);
                out.push_back(to_base64[b4_2]);
                out.push_back(to_base64[b4_3]);
            }
            
            /// Replace data that is invalid (always as many as there are missing bytes)
            for (std::size_t idx = 0; idx != missing; ++idx) {
                out[ret_size - idx - 1] = '=';
            }
        }
        
        std::string encode(byte const* buf, std::size_t bufLen) {
            std::string out;
            encode(out, buf, bufLen);
            return out;
        }
        
        std::string encode(char const* cstring, std::size_t len) {
            std::string out;
            encode(out, reinterpret_cast<byte const*>(cstring), len);
            return out;
        }
        
        std::string encode(char const* cstring) {
            std::size_t len = std::strlen(cstring);
            std::string out;
            encode(out, reinterpret_cast<byte const*>(cstring), len);
            return out;
        }
        
        template <typename OutType>
        void decode_any(OutType& out, std::string const& in) {
            using value_t = typename OutType::value_type;
            
            /// Make sure the *intended* string length is a multiple of 4
            std::size_t encoded_size = in.size();
            while ((encoded_size % 4) != 0) { ++encoded_size; }
            
            const std::size_t N = in.size();
            out.clear();
            out.reserve(3 * encoded_size / 4);
            
            for (std::size_t idx = 0; idx < encoded_size; idx += 4) {
                /// Note: 'z' == 122
                // Get values for each group of four base 64 characters
                const byte b4_0 = (              in[idx+0] <= 'z') ? from_base64[static_cast<byte>(in[idx+0])] : 0xff;
                const byte b4_1 = (idx+1 < N and in[idx+1] <= 'z') ? from_base64[static_cast<byte>(in[idx+1])] : 0xff;
                const byte b4_2 = (idx+2 < N and in[idx+2] <= 'z') ? from_base64[static_cast<byte>(in[idx+2])] : 0xff;
                const byte b4_3 = (idx+3 < N and in[idx+3] <= 'z') ? from_base64[static_cast<byte>(in[idx+3])] : 0xff;
                
                /// Transform into a group of three bytes
                const byte b3_0 = ((b4_0 & 0x3f) << 2) + ((b4_1 & 0x30) >> 4);
                const byte b3_1 = ((b4_1 & 0x0f) << 4) + ((b4_2 & 0x3c) >> 2);
                const byte b3_2 = ((b4_2 & 0x03) << 6) + ((b4_3 & 0x3f) >> 0);
                
                /// Add the byte to the return value if it isn't part of an '=' character (indicated by 0xff)
                if (b4_1 != 0xff) { out.push_back(static_cast<value_t>(b3_0)); }
                if (b4_2 != 0xff) { out.push_back(static_cast<value_t>(b3_1)); }
                if (b4_3 != 0xff) { out.push_back(static_cast<value_t>(b3_2)); }
            }
        }
        
        void decode(std::string& out, std::string const& encoded_string) {
            decode_any(out, encoded_string);
        }
        
        void decode(bytevec_t& out, std::string const& encoded_string) {
            decode_any(out, encoded_string);
        }
        
        charptr_t decode(std::string const& encoded_string) {
            /// ::strdup((char const*)out.data());
            bytevec_t out;
            decode_any(out, encoded_string);
            charptr_t outout = std::make_unique<char[]>(out.size());
            std::memcpy(outout.get(), out.data(), out.size());
            return outout;
        }
        
    }
    
}