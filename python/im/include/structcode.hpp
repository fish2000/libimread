#ifndef PyImgC_STRUCTCODE_H
#define PyImgC_STRUCTCODE_H

#include <cstdio>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace structcode {
    
    using stringmap_t = std::map<std::string, std::string>;
    using stringvec_t = std::vector<std::string>;
    
    struct structcodemaps {
        
        static stringmap_t init_byteorder() {
            stringmap_t _byteorder_map = {
                {"@", "="},
                {"=", "="},
                {"<", "<"},
                {">", ">"},
                {"^", "="},
                {"!", ">"},
            };
            return _byteorder_map;
        }
        
        static stringmap_t init_native() {
            stringmap_t _native_map = {
                {"?", "?"},
                {"b", "b"},
                {"B", "B"},
                {"h", "h"},
                {"H", "H"},
                {"i", "i"},
                {"I", "I"},
                {"l", "l"},
                {"L", "L"},
                {"q", "q"},
                {"Q", "Q"},
                {"e", "e"},
                {"f", "f"},
                {"d", "d"},
                {"g", "g"}, 
                {"Zf", "F"},
                {"Zd", "D"},
                {"Zg", "G"},
                {"s", "S"},
                {"w", "U"},
                {"O", "O"},
                {"x", "V"}, /// padding
            };
            return _native_map;
        }
        
        static stringmap_t init_standard() {
            stringmap_t _standard_map = {
                {"?", "?"},
                {"b", "b"},
                {"B", "B"},
                {"h", "i2"},
                {"H", "u2"},
                {"i", "i4"},
                {"I", "u4"},
                {"l", "i4"},
                {"L", "u4"},
                {"q", "i8"},
                {"Q", "u8"},
                {"e", "f2"},
                {"f", "f"},
                {"d", "d"},
                {"Zf", "F"},
                {"Zd", "D"},
                {"s", "S"},
                {"w", "U"},
                {"O", "O"},
                {"x", "V"}, /// padding
            };
            return _standard_map;
        }
        
        static const stringmap_t byteorder;
        static const stringmap_t native;
        static const stringmap_t standard;
    };
    
    struct field_namer {
        int idx;
        stringvec_t field_name_vector;
        field_namer();
        int next();
        void add(std::string const&);
        bool has(std::string const&);
        std::string operator()();
    };
    
    using shapevec_t = std::vector<int>;
    using structcode_t = std::vector<std::pair<std::string, std::string>>;
    
    using parse_result_t = std::tuple<std::string, stringvec_t, structcode_t>;
    
    shapevec_t parse_shape(std::string shapecode);
    parse_result_t parse(std::string structcode, bool toplevel=true);

} /// namespace structcode

#endif