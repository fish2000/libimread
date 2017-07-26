/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/options.hh>

namespace im {
    
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
