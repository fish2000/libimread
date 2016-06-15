#ifndef PyImgC_STRUCTCODE_H
#define PyImgC_STRUCTCODE_H

#include <string>
#include <utility>
#include <vector>
#include <map>

namespace structcode {
    
    using stringmap_t = std::map<std::string, std::string>;
    using stringvec_t = std::vector<std::string>;
    
    struct structcodemaps {
        
        static stringmap_t init_byteorder();
        static stringmap_t init_native();
        static stringmap_t init_standard();
        
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