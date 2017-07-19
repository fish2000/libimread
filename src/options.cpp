/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/options.hh>

namespace im {
    
    
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
    
    Options Options::parse(char const* json) {
        return parse(std::string(json));
    }
    
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
    
    OptionsList OptionsList::parse(char const* json) {
        return parse(std::string(json));
    }
    
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
