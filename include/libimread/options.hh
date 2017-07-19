/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef IMREAD_OPTIONS_HH_
#define IMREAD_OPTIONS_HH_

#include <string>
#include <sstream>
#include <libimread/ext/JSON/json11.h>

namespace im {
    
    struct Options : public Json {
        
        public:
            using Json::set;
            using Json::null;
            using Json::undefined;
        
        public:
            Options();
            Options(Json const&);
            Options(Json&&) noexcept;
            Options(std::istream& is, bool full = true);
            virtual ~Options();
            
            template <typename ConvertibleType,
                      typename = decltype(&ConvertibleType::to_json)>
            Options(ConvertibleType const& convertible)
                :Json(convertible.to_json())
                {}
        
        public:
            static Options parse(std::string const&);
            static Options parse(char const*);
            
    };
    
    struct OptionsList : public Json {
        
        public:
            using Json::set;
            using Json::null;
            using Json::undefined;
        
        public:
            OptionsList();
            OptionsList(Json const& other);
            OptionsList(Json&& other) noexcept;
            OptionsList(std::istream& is, bool full = true);
            virtual ~OptionsList();
            
            template <typename ConvertibleType,
                      typename = decltype(&ConvertibleType::to_json)>
            OptionsList(ConvertibleType const& convertible)
                :Json(convertible.to_json())
                {}
        
        public:
            static OptionsList parse(std::string const&);
            static OptionsList parse(char const*);
    
    };
    
    std::string get_optional_string(Options const& opts,  std::string const& key);
    const char* get_optional_cstring(Options const& opts, std::string const& key);
            int get_optional_int(Options const& opts,     std::string const& key,   int const default_value = 0);
           bool get_optional_bool(Options const& opts,    std::string const& key,  bool const default_value = false);
    
}

#endif /// IMREAD_OPTIONS_HH_