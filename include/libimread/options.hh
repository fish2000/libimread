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
    };
    
    std::string get_optional_string(const options_map &opts,  const std::string key);
    const char *get_optional_cstring(const options_map &opts, const std::string key);
    
    int get_optional_int(const options_map &opts,   const std::string key,
                                                    const int def);
    bool get_optional_bool(const options_map &opts, const std::string key,
                                                    const int def);
    
}

#endif /// IMREAD_OPTIONS_HH