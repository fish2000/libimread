/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <regex>
#include <utility>

#include <libimread/libimread.hpp>
#include <libimread/options.hh>

namespace im {
    
    using patternmap_t = std::unordered_map<std::string, std::regex>;
    
    #pragma mark -
    #pragma mark method implementations for im::Options
    
    Options::Options()
        :Json()
        { mkobject(); }
    
    Options::Options(Json const& other)
        :Json(other)
        {}
    Options::Options(Json&& other) noexcept
        :Json(std::move(other))
        {}
    
    Options::Options(std::istream& is, bool full)
        :Json(is, full)
        {}
    
    Options::Options(stringpair_init_t stringpair_init)
        :Json(Json::jsonmap_t(stringpair_init.begin(),
                              stringpair_init.end()))
        {}
    
    Options::~Options() {}
    
    Options Options::parse(std::string const& str) {
        std::istringstream is(str);
        Options parsed(is);
        if (is.peek() == std::char_traits<char>::eof()) { return parsed; }
        while (std::isspace(is.get()))
            /* skip */;
        if (is.eof()) { return parsed; }
        throw Json::parse_error("JSON im::Options format error", is);
    }
    
    bool Options::set(std::string const& key, std::string const& value) {
        cache[key] = value;
        Json::set(key, value);
        return Json::has(key);
    }
    
    bool Options::set(std::string const& key, Json const& value) {
        cache[key] = std::string(value);
        Json::set(key, value);
        return Json::has(key);
    }
    
    std::string& Options::get(std::string const& key) {
        if (cache.find(key) != cache.end()) { return cache[key]; }
        cache[key] = Json::cast<std::string>(key);
        return cache[key];
    }
    
    std::string const& Options::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) { return cache[key]; }
        cache[key] = Json::cast<std::string>(key);
        return cache[key];
    }
    
    bool Options::has(std::string const& key) const {
        return cache.find(key) != cache.end() || Json::has(key);
    }
    
    bool Options::del(std::string const& key) {
        if (cache.find(key) != cache.end()) { cache.erase(key); }
        return Json::remove(key);
    }
    
    // Json         update(Json const& other) const { return Json::update(other); }
    // Json         pop(std::string const& key) { return Json::pop(key); }
    // Json         pop(std::string const& key, Json const& default_value) { return Json::pop(key, default_value); }
    
    std::size_t Options::count() const { return Json::size(); }
    stringvec_t Options::list() const { return Json::keys(); }
    
    std::size_t Options::count(std::string const& prefix,
                               std::string const& separator) const {
        /// return the count of how many keys have this prefix:
        std::regex prefix_re("^" + prefix + separator, std::regex::extended);
        stringvec_t keys = Options::list();
        return std::count_if(keys.begin(), keys.end(),
                         [&](std::string const& key) { return std::regex_search(key, prefix_re); });
    }
    
    std::size_t Options::prefixcount(std::string const& prefix,
                                     std::string const& separator) const {
        /// convenience function to call Options::count(prefix, separator)
        /// with “the default separator” which is ":" and/or ':', depending:
        return Options::count(prefix, separator);
    }
    
    prefixset_t Options::prefixset(std::string const& separator) const {
        stringvec_t keys = Options::list();
        stringvec_t prefixvec;
        prefixset_t prefixes;
        
        /// fill a vector with prefixed keys:
        prefixvec.reserve(keys.size());
        std::copy_if(keys.begin(), keys.end(),
                     std::back_inserter(prefixvec),
                 [&](std::string const& s) { return bool(s.find(separator[0]) != std::string::npos); });
        
        /// cut the strings in the vector down to just the prefix:
        std::transform(prefixvec.begin(), prefixvec.end(),
                       prefixvec.begin(),
                   [&](std::string const& s) { return s.substr(0, s.find_first_of(separator[0])); });
        
        /// uniquify the contents of the prefix string vector:
        std::sort(prefixvec.begin(), prefixvec.end());
        auto last = std::unique(prefixvec.begin(), prefixvec.end());
        prefixvec.erase(last, prefixvec.end());
        
        /// copy the unique prefixes into a set for output and return it:
        prefixes.reserve(prefixvec.size());
        std::copy(prefixvec.begin(),     prefixvec.end(),
                  std::inserter(prefixes, prefixes.end()));
        return prefixes;
    }
    
    prefixgram_t Options::prefixgram(std::string const& separator) const {
        prefixset_t  prefixes = Options::prefixset(separator);
        patternmap_t patterns;
        prefixgram_t prefixgram;
        
        /// fill a map with regex patterns, one pattern per prefix:
        patterns.reserve(prefixes.size());
        std::transform(prefixes.begin(),       prefixes.end(),
                       std::inserter(patterns, patterns.end()),
                   [&](std::string const& s) { return std::make_pair(s, std::regex("^" + s + separator,
                                                                        std::regex::extended)); });
        
        /// count each pattern’s matches against a string vector
        /// containing a set of all keys, using a prefix histogram:
        stringvec_t keys = Options::list();
        prefixgram.reserve(patterns.size());
        std::for_each(patterns.begin(),
                      patterns.end(),
                  [&](auto const& kv) {
                       prefixgram[kv.first] = std::count_if(keys.begin(), keys.end(),
                                                        [&](std::string const& s) { 
                return std::regex_search(s, kv.second);
            });
        });
        
        /// return the prefix histogram (née “prefixgram”):
        return prefixgram;
    }
    
    #pragma mark -
    #pragma mark method implementations for im::OptionsList
    
    OptionsList::OptionsList()
        :Json()
        { mkarray(); }
    
    OptionsList::OptionsList(Json const& other)
        :Json(other)
        {}
    
    OptionsList::OptionsList(Json&& other) noexcept
        :Json(std::move(other))
        {}
    
    OptionsList::OptionsList(std::istream& is, bool full)
        :Json(is, full)
        {}
    
    OptionsList::OptionsList(string_init_t string_init)
        :Json(Json::jsonvec_t(string_init.begin(),
                              string_init.end()))
        {}
    
    OptionsList::~OptionsList() {}
    
    OptionsList OptionsList::parse(std::string const& str) {
        std::istringstream is(str);
        OptionsList parsed(is);
        if (is.peek() == std::char_traits<char>::eof()) { return parsed; }
        while (std::isspace(is.get()))
            /* skip */;
        if (is.eof()) { return parsed; }
        throw Json::parse_error("JSON im::OptionsList format error", is);
    }
    
    #pragma mark -
    #pragma mark im::get_optional_{string,cstring,int,bool} legacy methods
    
    std::string           get_optional_string(Options const& opts,
                                              std::string const& key) {
        return opts.cast<std::string>(key, "");
    }
    
    char const*           get_optional_cstring(Options const& opts,
                                               std::string const& key) {
        return opts.cast<char const*>(key, "");
    }
    
    int                   get_optional_int(Options const& opts,
                                           std::string const& key,
                                           int const default_value) {
        return opts.cast<int>(key, default_value);
    }
    
    bool                  get_optional_bool(Options const& opts,
                                            std::string const& key,
                                            bool const default_value) {
        return opts.cast<int>(key, static_cast<int>(default_value));
    }
    
}
