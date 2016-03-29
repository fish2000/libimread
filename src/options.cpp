/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/options.hh>

namespace im {
    
    options_map options_map::parse(std::string const& str) {
        std::istringstream is(str);
        options_map parsed(is);
        if (is.peek() == std::char_traits<char>::eof()) { return parsed; }
        while (std::isspace(is.get()))
            /* skip */;
        if (is.eof()) { return parsed; }
        throw Json::parse_error("JSON format error", is);
    }
    
    
    std::string           get_optional_string(options_map const& opts,
                                              std::string const& key) {
        return opts.cast<std::string>(key, "");
    }
    
    char const*           get_optional_cstring(options_map const& opts,
                                               std::string const& key) {
        return opts.cast<char const*>(key, "");
    }
    
    int                   get_optional_int(options_map const& opts,
                                           std::string const& key,
                                           int const default_value) {
        return opts.cast<int>(key, default_value);
    }
    
    bool                  get_optional_bool(options_map const& opts,
                                            std::string const& key,
                                            bool const default_value) {
        return opts.cast<int>(key, static_cast<int>(default_value));
    }
    
}
