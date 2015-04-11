// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <cstdio>
#include <string>
#include <type_traits>
#include <iostream>
#include <exception>

#include <libimread/libimread.hpp>

    
#ifndef WTF
#define WTF(...) wtf(__FILE__, __LINE__, #__VA_ARGS__)
#endif

namespace im {
    
    template <bool B, typename T = void>
    using disable_if = std::enable_if<!B, T>;
    template <bool B, typename T = void>
    using disable_if_t = std::enable_if_t<!B, T>;
    
    template <typename S>
    typename std::enable_if_t<std::is_arithmetic<S>::value, std::string>
        stringify(S s) { return std::to_string(s); }
    
    // template <typename S,
    //           typename std::enable_if_t<std::is_convertible<S, std::string>::value>* = nullptr>
    // std::string stringify(S s) { return s; }
    
    // template <typename S,
    //           typename std::enable_if<std::is_convertible<S, std::string>::value>::type* = nullptr>
    // const std::string &stringify(const S &s) { return s; }
    
    // template <typename S>
    // typename std::enable_if_t<std::is_constructible<std::string, S>::value, std::string>
    //     stringify(S s) { return std::string(const_cast<S>(s)); }
    
    template <typename S>
    typename std::enable_if_t<std::is_constructible<std::string, S>::value, const std::string>
        stringify(S const& s) { return std::string(s); }
    
    template <typename ...Args>
    std::string stringmerge(const Args& ...args) {
        /// adapted from http://stackoverflow.com/a/21806609/298171
        std::string out;
        int unpack[]{ 0, (out += "\t" + im::stringify<Args>(args) + "\n", 0)... };
        static_cast<void>(unpack); /// avoid "unused variable" warnings
        return out;
    }
    
    template <typename ...Args>
    void wtf(const char *file, int line, Args&& ...args) {
        
        std::cerr  << "(WTF!!!) [ "
                   << im::stringify(file)
                   << " : " << im::stringify(line) << " ]:\n"
                   << im::stringmerge(std::forward<Args>(args)...)
                   << "\n ";
    }

#ifndef _ASSERT
#define _ASSERT(condition, ...)                                                 \
    if (!(condition)) {                                                         \
        fprintf(stderr, __VA_ARGS__);                                           \
        exit(-1);                                                               \
    }
#endif /// _ASSERT

#define DECLARE_IMREAD_ERROR_TYPE(TypeName, DefaultMsg)                         \
    struct TypeName : std::exception {                                          \
        TypeName(const char *e)                                                 \
            :w(e)                                                               \
            { }                                                                 \
        TypeName(std::string e)                                                 \
            :w(e)                                                               \
            { }                                                                 \
        TypeName()                                                              \
            :w(DefaultMsg)                                                      \
            { }                                                                 \
        ~TypeName() throw() { }                                                 \
                                                                                \
        const char* what() const throw() { return w.c_str(); }                  \
        std::string w;                                                          \
    };

DECLARE_IMREAD_ERROR_TYPE(CannotReadError, "Read Error");
DECLARE_IMREAD_ERROR_TYPE(CannotWriteError, "Write Error");
DECLARE_IMREAD_ERROR_TYPE(NotImplementedError, "Not Implemented");
DECLARE_IMREAD_ERROR_TYPE(ProgrammingError, "Programming Error");
DECLARE_IMREAD_ERROR_TYPE(OptionsError, "Options Error");
DECLARE_IMREAD_ERROR_TYPE(WriteOptionsError, "Write Options Error");

}

#endif // LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
