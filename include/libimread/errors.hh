// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>
#include <iostream>
#include <exception>

#include <libimread/libimread.hpp>
#include <libimread/ansicolor.hh>

namespace im {
    
    enum class Output {
        DEFAULT, WTF, OMG, SRSLY
    };
    
    template <bool B, typename T = void>
    using disable_if = std::enable_if<!B, T>;
    
    template <bool B, typename T = void>
    using disable_if_t = std::enable_if_t<!B, T>;
    
    /// apply_impl() and apply() courtesy of http://stackoverflow.com/a/19060157/298171
    template<typename F, typename Tuple, std::size_t ...I> inline
    auto apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
        return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
    }
    template<typename F, typename Tuple> inline
    auto apply(F&& f, Tuple&& t) {
        using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
        return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices());
    }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_arithmetic<S>::value, std::string>
        stringify(S s) { return std::to_string(s); }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_constructible<std::string, S>::value, const std::string>
        stringify(S const& s) { return std::string(s); }
    
    template <typename S, typename ...Args> inline
    typename std::enable_if_t<std::is_constructible<std::string, S>::value && (sizeof...(Args) != 0), const std::string>
        stringify(S const& s, Args ...args) {
            /// adapted from http://stackoverflow.com/a/26197300/298171
            char b; std::string fmt(s);
            unsigned required = std::snprintf(&b, 0, fmt.c_str(), args...) + 1;
            char bytes[required];
            std::snprintf(bytes, required, fmt.c_str(), args...);
            return std::string(bytes);
        }
    
    struct stringifier {
        template <typename S, typename ...Args>
        const std::string operator()(S&& s, Args&& ...args) const {
            return im::stringify(
                std::forward<S>(s),
                std::forward<Args>(args)...);
        }
    };
    
    template <typename S> inline
    typename std::enable_if_t<std::tuple_size<S>::value != 0, const std::string>
        stringify(S const& s) { return im::apply(im::stringifier(), s); }
    
    template <typename ...Args> inline
    std::string stringmerge(const Args& ...args) {
        /// adapted from http://stackoverflow.com/a/21806609/298171
        std::string out;
        int unpack[] __attribute__((unused)) { 0, 
            (out += "\t" + im::stringify<Args>(args) + "\n", 0)...
        };
        return out;
    }
    
    template <typename ...Args> inline
    void __attribute__((nonnull (1, 3)))
         srsly(const char *title, const ansi::ANSI color,
               const char *file, int line, Args&& ...args) {
        std::cerr  << color << im::stringify(title) << ansi::reset
          << " [ " << ansi::yellow << im::stringify(file) << ansi::reset
          << " : " << ansi::red << im::stringify(line) << ansi::reset
          << " ]:" << std::endl
                   << im::stringmerge(std::forward<Args>(args)...)
                   << std::endl;
    }

#ifndef FF
#define FF(...) std::forward_as_tuple(__VA_ARGS__)
#endif

#ifndef WTF
#define WTF(...)                                                                \
    im::srsly("(WTF!!!)",                                                       \
        ansi::color::idx.at("lightred"),                                        \
        __FILE__, __LINE__, __VA_ARGS__)
#endif /// WTF

#ifndef _ASSERT
#define _ASSERT(condition, ...)                                                 \
    if (!(condition)) {                                                         \
        im::srsly("(ASSERT FAILURE) [ " #condition " ]",                        \
            ansi::color::idx.at("lightyellow"),                                 \
            __FILE__, __LINE__, __VA_ARGS__);                                   \
        exit(-1);                                                               \
    }
#endif /// _ASSERT

#ifndef DECLARE_IMREAD_ERROR_TYPE
#define DECLARE_IMREAD_ERROR_TYPE(TypeName, DefaultMsg)                         \
    struct TypeName : std::exception {                                          \
        template <typename S>                                                   \
        TypeName(S s)                                                           \
            :w(im::stringify(s))                                                \
            { }                                                                 \
        template <typename S, typename ...Args>                                 \
        TypeName(S s, Args&& ...args)                                           \
            :w(im::stringify(s) + im::stringmerge(std::forward<Args>(args)...)) \
            { }                                                                 \
        TypeName()                                                              \
            :w(DefaultMsg)                                                      \
            { }                                                                 \
        ~TypeName() throw() { }                                                 \
                                                                                \
        const char* what() const throw() { return w.c_str(); }                  \
        std::string w;                                                          \
    };
#endif /// DECLARE_IMREAD_ERROR_TYPE

DECLARE_IMREAD_ERROR_TYPE(CannotReadError, "Read Error");
DECLARE_IMREAD_ERROR_TYPE(CannotWriteError, "Write Error");
DECLARE_IMREAD_ERROR_TYPE(NotImplementedError, "Not Implemented");
DECLARE_IMREAD_ERROR_TYPE(ProgrammingError, "Programming Error");
DECLARE_IMREAD_ERROR_TYPE(OptionsError, "Options Error");
DECLARE_IMREAD_ERROR_TYPE(WriteOptionsError, "Write Options Error");

}

#endif // LPC_ERRORS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
