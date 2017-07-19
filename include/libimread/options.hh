/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef IMREAD_OPTIONS_HH
#define IMREAD_OPTIONS_HH

#include <string>
#include <sstream>
#include <libimread/ext/JSON/json11.h>

namespace im {
    
    struct Options : public Json {
        public:
            using Json::set;
            using Json::null;
            using Json::undefined;
            
            Options()
                :Json()
                { mkobject(); }
            
            Options(Json const& other)
                :Json(other)
                {}
            Options(Json&& other) noexcept
                :Json(other)
                {}
            
            Options(std::istream& is, bool full = true)
                :Json(is, full)
                {}
            
            template <typename ConvertibleType,
                      typename = decltype(&ConvertibleType::to_json)>
            Options(ConvertibleType const& convertible)
                :Json(convertible.to_json())
                {}
            
            static Options parse(std::string const&);
            static Options parse(char const* json) { return parse(std::string(json)); }
            
    };
    
    struct OptionsList : public Json {
        public:
            using Json::set;
            using Json::null;
            using Json::undefined;
            
            OptionsList()
                :Json()
                { mkarray(); }
            
            OptionsList(Json const& other)
                :Json(other)
                {}
            OptionsList(Json&& other) noexcept
                :Json(other)
                {}
            
            template <typename ConvertibleType,
                      typename = decltype(&ConvertibleType::to_json)>
            OptionsList(ConvertibleType const& convertible)
                :Json(convertible.to_json())
                {}
    };
    
    std::string get_optional_string(Options const& opts,  std::string const& key);
    const char* get_optional_cstring(Options const& opts, std::string const& key);
    
    int get_optional_int(Options const& opts,
                         std::string const& key,    int const default_value = 0);
    bool get_optional_bool(Options const& opts,
                           std::string const& key, bool const default_value = false);
    
}

#endif /// IMREAD_OPTIONS_HH