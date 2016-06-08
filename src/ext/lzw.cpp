/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/ext/lzw.hh>
#include <libimread/errors.hh>

namespace im {
    
    namespace lzw {
    
        code_stream::code_stream(byte* buf, unsigned long len)
            :buf_(buf)
            ,byte_pos_(0)
            ,bit_pos_(0)
            ,len_(len)
            {}
        
        unsigned short code_stream::getbit() {
            byte val = buf_[byte_pos_];
            unsigned short res = (val & (1 << (8-bit_pos_)));
            ++bit_pos_;
            if (bit_pos_ == 8) {
                bit_pos_ = 0;
                ++byte_pos_;
                if (byte_pos_ > len_) {
                    imread_raise(CannotReadError,
                        "Unexpected End of File");
                }
            }
            return res;
        }
        
        unsigned short code_stream::get(int nbits) {
            unsigned short res = 0;
            for (int idx = 0; idx != nbits; ++idx) {
                res <<= 1;
                res |= getbit();
            }
            return res;
        }
        
        namespace detail {
            
            std::string table_at(stringvec_t const& table, unsigned short idx) {
                if (idx < 256) {
                    std::string res("0");
                    res[0] = (char)idx;
                    return res;
                }
                return table.at(idx - 258);
            }
            
            void write_string(bytevec_t& output, std::string const& s) {
                output.insert(output.end(),
                              s.begin(), s.end());
            }
            
        } /* namespace detail */
        
        bytevec_t decode(void* buf, unsigned long len) {
            stringvec_t table;
            bytevec_t output;
            code_stream st(static_cast<byte*>(buf), len);
            
            int nbits = 9;
            unsigned short old_code = 0;
            const short ClearCode = 256;
            const short EoiCode = 257;
            
            while (true) {
                const short code = st.get(nbits);
                if (code == EoiCode) { break; }
                
                if (code == ClearCode) {
                    table.clear();
                    nbits = 9;
                    const short next_code = st.get(nbits);
                    if (next_code == EoiCode) break;
                    detail::write_string(output, table[next_code]);
                    old_code = next_code;
                
                } else if (code < 256 || (code - 258) < short(table.size())) {
                    detail::write_string(output,
                        detail::table_at(table, code));
                    table.push_back(
                        detail::table_at(table, old_code) +
                        detail::table_at(table, code)[0]);
                    old_code = code;
                
                } else {
                    std::string out_string = detail::table_at(table, old_code) +
                                             detail::table_at(table, old_code)[0];
                    detail::write_string(output, out_string);
                    table.push_back(out_string);
                    if (table.size() == ( 512-258)) nbits = 10;
                    if (table.size() == (1024-258)) nbits = 11;
                    if (table.size() == (2048-258)) nbits = 12;
                    old_code = code;
                }
           }
           return output;
        }
    
    } /* namespace lzw */
    
} /* namespace im */

