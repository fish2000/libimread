/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/ansicolor.hh>

#ifndef IM_COLOR_TRACE
#define IM_COLOR_TRACE 0
#endif

namespace ansi {
        
    ANSI::ANSI(code_t c)
        :code(c)
        {}
    
    std::ostream& operator<<(std::ostream& os, ANSI const& a) {
        #if IM_COLOR_TRACE == 1
            return os << "\033[" << static_cast<std::size_t>(a.code) << "m";
        #else
            return os;
        #endif
    }
    
    std::ostream& operator<<(std::ostream& os, codedtext_t const& tuple) {
        #if IM_COLOR_TRACE == 1
            return os << std::get<0>(tuple) << std::get<1>(tuple) << std::get<2>(tuple);
        #else
            return os << std::get<1>(tuple);
        #endif
    }
    
    std::string ANSI::str() const {
        #if IM_COLOR_TRACE == 1
            return "\033[" + std::to_string(
                             static_cast<std::size_t>(code)) + "m";
        #else
            return "";
        #endif
    }
    
    std::string ANSI::str(std::string const& colored) const {
        #if IM_COLOR_TRACE == 1
            return "\033[" + std::to_string(
                             static_cast<std::size_t>(code)) + "m" +
                    colored + "\033[0m";
        #else
            return colored;
        #endif
    }
    
    char const* ANSI::c_str() const {
        #if IM_COLOR_TRACE == 1
            return str().c_str();
        #else
            return "";
        #endif
    }
    
    ANSI::operator std::string() const { return str(); }
    ANSI::operator const char*() const { return str().c_str(); }
    
    std::size_t ANSI::value() const {
        return static_cast<std::size_t>(code);
    }
    
    codeformat_t ANSI::format() const {
        codeformat_t codeformat;
        codeformat.first = "\033[" + std::to_string(
                         static_cast<std::size_t>(code))
                                   + "m";
        codeformat.second = "\033[0m";
        return codeformat;
    }
    
    codedtext_t ANSI::formatted(std::string const& text) const {
        codeformat_t codeformat = format();
        return {
            std::move(codeformat.first), text,
            std::move(codeformat.second)
        };
    }
    
    std::string ANSI::operator()(std::string const& text) const {
        static const codeformat_t codeformat = format();
        return codeformat.first + text + codeformat.second;
    }
    
}
