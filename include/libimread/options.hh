/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef IMREAD_OPTIONS_HH
#define IMREAD_OPTIONS_HH

#include <string>
#include <sstream>
#include <libimread/ext/JSON/json11.h>

namespace im {
    
    struct options_map : public Json {
        public:
            using Json::set;
            using Json::null;
            using Json::undefined;
            
            options_map()
                :Json()
                { mkobject(); }
            
            options_map(Json const& other)
                :Json(other)
                {}
            options_map(Json&& other) noexcept
                :Json(other)
                {}
            
            options_map(std::istream& is, bool full = true)
                :Json(is, full)
                {}
            
            static options_map parse(std::string const&);
            static options_map parse(char const* json) { return parse(std::string(json)); }
            
    };
    
    struct options_list : public Json {
        public:
            using Json::set;
            using Json::null;
            using Json::undefined;
            
            options_list()
                :Json()
                { mkarray(); }
    };
    
    std::string get_optional_string(options_map const& opts,  std::string const& key);
    const char* get_optional_cstring(options_map const& opts, std::string const& key);
    
    int get_optional_int(options_map const& opts,
                         std::string const& key,    int const default_value = 0);
    bool get_optional_bool(options_map const& opts,
                           std::string const& key, bool const default_value = false);
    
}

#endif /// IMREAD_OPTIONS_HH