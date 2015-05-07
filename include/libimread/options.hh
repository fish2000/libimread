/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef IMREAD_OPTIONS_HH
#define IMREAD_OPTIONS_HH

#include <libimread/ext/JSON/json11.h>
#include <string>
#include <cstring>
#include <tuple>
#include <unordered_map>
#include <type_traits>

namespace im {
    
    using options_map = Json;
    
    inline std::string           get_optional_string(const options_map &opts,
                                                     const std::string key) {
        return opts.has(key) ? std::string(opts.get(key)) : std::string("");
    }
    
    inline const char           *get_optional_cstring(const options_map &opts,
                                                      const std::string key) {
        return get_optional_string(opts, key).c_str();
    }
    
    inline int                   get_optional_int(const options_map &opts,
                                                  const std::string key,
                                                  const int def) {
        return opts.has(key) ? int(opts.get(key)) : def;
    }
    
    inline bool                  get_optional_bool(const options_map &opts,
                                                   const std::string key,
                                                   const int def) {
        return get_optional_int(opts, key, def);
    }
    
}

#endif /// IMREAD_OPTIONS_HH