/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef IMREAD_OPTIONS_HH
#define IMREAD_OPTIONS_HH

#include <string>

#include <libimread/ext/iod.hh>
#include <libimread/ext/JSON/json11.h>

namespace im {
    
    struct options_map : public Json {
        public:
            using Json::set;
            
            options_map()
                :Json()
                { set("metadata", ""); }
            
            options_map(std::istream& is, bool full=true)
                :Json(is, full)
                {}
            
            static options_map parse(const std::string&);
            static options_map parse(const char* json) { return parse(std::string(json)); }
            
    };
    
    std::string get_optional_string(const options_map& opts,  const std::string& key);
    const char *get_optional_cstring(const options_map& opts, const std::string& key);
    
    int get_optional_int(const options_map& opts,   const std::string& key,
                                                    const int default_value);
    bool get_optional_bool(const options_map& opts, const std::string& key,
                                                    const bool default_value);
    
}

#endif /// IMREAD_OPTIONS_HH