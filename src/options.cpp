
#include <libimread/libimread.hpp>
#include <libimread/options.hh>

namespace im {
    
    std::string           get_optional_string(const options_map &opts,
                                                     const std::string key) {
        return opts.has(key) ? std::string(opts.get(key)) : std::string("");
    }
    
    const char           *get_optional_cstring(const options_map &opts,
                                                      const std::string key) {
        return get_optional_string(opts, key).c_str();
    }
    
    int                   get_optional_int(const options_map &opts,
                                                  const std::string key,
                                                  const int def) {
        return opts.has(key) ? int(opts.get(key)) : def;
    }
    
    bool                  get_optional_bool(const options_map &opts,
                                                   const std::string key,
                                                   const int def) {
        return get_optional_int(opts, key, def);
    }
    
}
