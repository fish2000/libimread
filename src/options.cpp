/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <string>

#include <libimread/libimread.hpp>
#include <libimread/ext/JSON/json11.h>
#include <libimread/options.hh>

namespace im {
    
    options_map options_map::parse(const std::string& str) {
        std::istringstream is(str);
        options_map parsed(is);
        if (is.peek() == std::char_traits<char>::eof()) { return parsed; }
        while (std::isspace(is.get()))
            /* skip */;
        if (is.eof()) { return parsed; }
        throw Json::parse_error("JSON format error", is);
    }
    
    
    std::string           get_optional_string(const options_map& opts,
                                              const std::string& key) {
        return opts.has(key) ? std::string(opts.get(key)) : std::string("");
    }
    
    const char           *get_optional_cstring(const options_map& opts,
                                               const std::string& key) {
        return get_optional_string(opts, key).c_str();
    }
    
    int                   get_optional_int(const options_map& opts,
                                           const std::string& key,
                                           const int default_value) {
        return opts.has(key) ? int(opts.get(key)) : default_value;
    }
    
    bool                  get_optional_bool(const options_map &opts,
                                            const std::string& key,
                                            const bool default_value) {
        return get_optional_int(opts, key,
                                static_cast<bool>(default_value));
    }
    
}
