/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/ansicolor.hh>

#ifndef IM_COLOR_TRACE
#define IM_COLOR_TRACE 0
#endif

namespace ansi {
    
    ANSI::ANSI(code c)
        :m_code(c)
        {}
    
    std::ostream& operator<<(std::ostream& os, ANSI const& a) {
        #if IM_COLOR_TRACE == 1
            return os << "\033[" << static_cast<std::size_t>(a.m_code) << "m";
        #else
            return os;
        #endif
    }
    
    std::string ANSI::str() const {
        #if IM_COLOR_TRACE == 1
            return "\033[" + std::to_string(
                             static_cast<std::size_t>(m_code)) + "m";
        #else
            return "";
        #endif
    }
    
    std::string ANSI::str(std::string const& colored) const {
        #if IM_COLOR_TRACE == 1
            return "\033[" + std::to_string(
                             static_cast<std::size_t>(m_code)) + "m" +
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
    
    std::size_t ANSI::code_value() const {
        return static_cast<std::size_t>(m_code);
    }
    
    ANSI::operator std::string() const { return str(); }
    ANSI::operator const char*() const { return str().c_str(); }
}
