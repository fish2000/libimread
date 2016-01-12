#ifndef PyImgC_STRUCTCODE_H
#define PyImgC_STRUCTCODE_H

#include <cstdio>
#include <iostream>
#include <sstream>
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

    //const stringmap_t structcodemaps::byteorder = structcodemaps::init_byteorder();
    //const stringmap_t structcodemaps::native = structcodemaps::init_native();
    //const stringmap_t structcodemaps::standard = structcodemaps::init_standard();

    struct field_namer {
        int idx;
        stringvec_t field_name_vector;
        field_namer():idx(0) {}
        int next() { return idx++; }
        void add(std::string const& field_name) { field_name_vector.push_back(field_name); }
        bool has(std::string const& field_name) {
            for (auto fn = std::begin(field_name_vector);
                      fn != std::end(field_name_vector); ++fn) {
                if (std::string(*fn) == field_name) {
                    return true;
                }
            }
            return false;
        }
        std::string operator()() {
            char str[5];
            while (true) {
                std::sprintf(str, "f%i", next());
                std::string dummy_name = std::string(str);
                if (!has(dummy_name)) {
                    add(dummy_name);
                    return dummy_name;
                }
            }
        }
    };
    
    using shapevec_t = std::vector<int>;
    using structcode_t = std::vector<std::pair<std::string, std::string>>;
    
    shapevec_t parse_shape(std::string shapecode);
    structcode_t parse(std::string structcode, bool toplevel=true);

} /// namespace structcode

#endif