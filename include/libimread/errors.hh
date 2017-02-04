/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ERRORS_HH_
#define LIBIMREAD_ERRORS_HH_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <functional>
#include <type_traits>
#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include <libimread/libimread.hpp>
#include <libimread/ansicolor.hh>

namespace im {
    
    template <bool B, typename T = void>
    using disable_if = std::enable_if<!B, T>;
    
    template <bool B, typename T = void>
    using disable_if_t = std::enable_if_t<!B, T>;
    
    /// apply_impl() and apply() courtesy of http://stackoverflow.com/a/19060157/298171
    template <typename F, typename Tuple, std::size_t ...I> inline
    auto apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
        return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
    }
    template <typename F, typename Tuple> inline
    auto apply(F&& f, Tuple&& t) {
        using Indices = std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>;
        return apply_impl(std::forward<F>(f), std::forward<Tuple>(t), Indices());
    }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_arithmetic<S>::value,
        const std::string>
        stringify(S s) { return std::to_string(s); }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_constructible<std::string, S>::value,
        const std::string>
        stringify(S&& s) { return std::string(std::forward<S>(s)); }
    
    template <typename S> inline
    typename std::enable_if_t<std::is_convertible<S, std::string>::value,
        const std::string>
        stringify(S const& s) { std::string out = s; return out; }
    
    template <typename S, typename ...Args> inline
    typename std::enable_if_t<std::is_constructible<std::string, S>::value && (sizeof...(Args) != 0),
        const std::string>
        stringify(S const& s, Args&& ...args) {
            /// adapted from http://stackoverflow.com/a/26197300/298171
            char b; const char* fmt(s);
            unsigned required = std::snprintf(&b, 0, fmt, std::forward<Args>(args)...) + 1;
            char bytes[required];
            std::snprintf(bytes, required, fmt,
						  std::forward<Args>(args)...);
            return std::string(bytes);
        }
    
    struct stringifier {
        constexpr stringifier() noexcept = default;
        stringifier(const stringifier&) noexcept {};
        template <typename S, typename ...Args> inline
        const std::string operator()(S&& s, Args&& ...args) const {
            return im::stringify(std::forward<S>(s),
                                 std::forward<Args>(args)...);
        }
    };
    
    template <typename S> inline
    typename std::enable_if_t<std::tuple_size<S>::value != 0,
        const std::string>
        stringify(S const& s) { return im::apply(im::stringifier(), s); }
    
    template <typename ...Args> inline
    std::string stringmerge(Args const& ...args) {
        /// adapted from http://stackoverflow.com/a/21806609/298171
        std::string out;
        unpack {
            (out += "\t" + im::stringify<Args>(args) + "\n", 0)...
        };
        return out;
    }
    
    template <typename ...Args> inline
    void __attribute__((nonnull (1, 3)))
         srsly(char const* title, ansi::ANSI const& color,
               char const* file, int line, Args&& ...args) {
        std::cerr  << color << im::stringify(title) << ansi::reset
          << " [ " << ansi::yellow << im::stringify(file) << ansi::reset
          << " : " << ansi::red << im::stringify(line) << ansi::reset
          << " ]:" << std::endl
                   << im::stringmerge(std::forward<Args>(args)...)
                   << std::endl;
    }
    
    template <typename ...Args> inline
    std::string __attribute__((nonnull (1, 3)))
         emerg(char const* title, ansi::ANSI const& color,
               char const* file, int line, Args&& ...args) {
        std::ostringstream errstream;
        errstream  << color << im::stringify(title) << ansi::reset
          << " [ " << ansi::yellow << im::stringify(file) << ansi::reset
          << " : " << ansi::red << im::stringify(line) << ansi::reset
          << " ]:" << std::endl
                   << im::stringmerge(std::forward<Args>(args)...)
                   << std::endl;
        return errstream.str();
    }
    
    template <typename ...Args> inline
    void norly(ansi::ANSI const& color, Args&& ...args) {
        std::cerr  << color 
                   << im::stringmerge(std::forward<Args>(args)...)
                   << ansi::reset << std::endl;
    }
    
#ifndef FF
#define FF(...) std::forward_as_tuple(__VA_ARGS__)
#endif /// FF

#ifndef FORSURE
#define FORSURE(...) im::norly(ansi::white, __VA_ARGS__)
#endif /// FORSURE

#ifndef WTF
#define WTF(...)                                                                        \
    im::srsly("\n(WTF!!!)",                                                             \
        ansi::lightred,                                                                 \
        __FILE__, __LINE__, __VA_ARGS__)
#endif /// WTF

#ifndef IM_VERBOSE
#define IM_VERBOSE 0
#endif

#if IM_VERBOSE == 1
/*
  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VERBOSE ERROR MACROS
  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
*/

#ifndef imread_assert
#define imread_assert(condition, ...)                                                   \
    if (!(condition)) {                                                                 \
        im::srsly("\n(ASSERT FAILURE) [ " ST(condition) " ]\n",                         \
            ansi::lightyellow, __FILE__, __LINE__,                                      \
                FF("\tIn function: %s%s%s",                                             \
                    ansi::bold.c_str(), __PRETTY_FUNCTION__, ansi::reset.c_str()),      \
                __VA_ARGS__);                                                           \
        std::exit(-1);                                                                  \
    }
#endif /// imread_assert

#ifndef imread_raise_default
#define imread_raise_default(Exception)                                                 \
    throw im::Exception(im::emerg("\n[ ERROR > " ST(Exception) " ]\n",                  \
        ansi::lightred, __FILE__, __LINE__,                                             \
            FF("\tIn function: %s%s%s",                                                 \
                ansi::bold.c_str(), __PRETTY_FUNCTION__, ansi::reset.c_str()),          \
            FF("\t%s", im::Exception::default_message)));
#endif /// imread_raise_default

#ifndef imread_raise
#define imread_raise(Exception, ...)                                                    \
    throw im::Exception(im::emerg("\n[ ERROR > " ST(Exception) " ]\n",                  \
        ansi::lightred, __FILE__, __LINE__,                                             \
            FF("\tIn function: %s%s%s",                                                 \
                ansi::bold.c_str(), __PRETTY_FUNCTION__, ansi::reset.c_str()),          \
            __VA_ARGS__));
#endif /// imread_raise

#else /// IM_VERBOSE == 0
/*
  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  >>>>>>>>>>>>>>> MILD-MANNERED, ECONOMICALLY-SPOKEN, GOOD-TURN-OF-PHRASE ERROR MACROS
  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
*/

#ifndef imread_assert
#define imread_assert(condition, ...)                                                   \
    if (!(condition)) {                                                                 \
        im::srsly("(ASSERT FAILURE) [ " ST(condition) " ]\n",                           \
            ansi::lightyellow, __FILE__, __LINE__,                                      \
                __VA_ARGS__);                                                           \
        std::exit(-1);                                                                  \
    }
#endif /// imread_assert

#ifndef imread_raise_default
#define imread_raise_default(Exception)                                                 \
    throw im::Exception("[ EXCEPTION (default) ]\n",                                    \
        FF("[ " ST(Exception) " > %s : %i ]", __FILE__, __LINE__),                      \
        FF("\t%s", im::Exception::default_message));
#endif /// imread_raise_default

#ifndef imread_raise
#define imread_raise(Exception, ...)                                                    \
    throw im::Exception("[ EXCEPTION ]\n",                                              \
        FF("[ " ST(Exception) " > %s : %i ]", __FILE__, __LINE__),                      \
        __VA_ARGS__);
#endif /// imread_raise

#endif /// IM_VERBOSE == 1

#ifndef DECLARE_IMREAD_ERROR_INNARDS
#define DECLARE_IMREAD_ERROR_INNARDS(TypeName, DefaultMsg, MsgLen)                      \
    constexpr static const char default_message[MsgLen] = ST(DefaultMsg);               \
    template <typename S>                                                               \
    TypeName(S s)                                                                       \
        :w(im::stringify(s))                                                            \
        { }                                                                             \
    template <typename S, typename ...Args>                                             \
    TypeName(S s, Args&& ...args)                                                       \
        :w(im::stringify(s) + im::stringmerge(std::forward<Args>(args)...))             \
        { }                                                                             \
    TypeName()                                                                          \
        :w(DefaultMsg)                                                                  \
        { }                                                                             \
    virtual ~TypeName() { }                                                             \
                                                                                        \
    const char* what() const noexcept { return w.c_str(); }                             \
    std::string w;
#endif /// DECLARE_IMREAD_ERROR_INNARDS

#ifndef DECLARE_IMREAD_ERROR_SUBTYPE_INNARDS
#define DECLARE_IMREAD_ERROR_SUBTYPE_INNARDS(TypeName, BaseTypeName, DefaultMsg, MsgLen)\
    static constexpr char default_message[MsgLen] = ST(DefaultMsg);                     \
    template <typename S>                                                               \
    TypeName(S s)                                                                       \
        :BaseTypeName(w)                                                                \
        ,w(im::stringify(s))                                                            \
        { }                                                                             \
    template <typename S, typename ...Args>                                             \
    TypeName(S s, Args&& ...args)                                                       \
        :BaseTypeName(w)                                                                \
        ,w(im::stringify(s) + im::stringmerge(std::forward<Args>(args)...))             \
        { }                                                                             \
    TypeName()                                                                          \
        :BaseTypeName(DefaultMsg), w(DefaultMsg)                                        \
        { }                                                                             \
    virtual ~TypeName() { }                                                             \
                                                                                        \
    const char* what() const noexcept { return w.c_str(); }                             \
    std::string w;
#endif /// DECLARE_IMREAD_ERROR_SUBTYPE_INNARDS

#ifndef DECLARE_IMREAD_ERROR_TYPE
#define DECLARE_IMREAD_ERROR_TYPE(TypeName, DefaultMsg)                                 \
    struct TypeName : public std::exception {                                           \
        public:                                                                         \
        DECLARE_IMREAD_ERROR_INNARDS(TypeName, DefaultMsg,                              \
            static_strlen(ST(DefaultMsg)));                                             \
    };
#endif /// DECLARE_IMREAD_ERROR_TYPE

#ifndef DECLARE_IMREAD_ERROR_SUBTYPE
#define DECLARE_IMREAD_ERROR_SUBTYPE(TypeName, BaseTypeName, DefaultMsg)                \
    struct TypeName : public BaseTypeName {                                             \
        public:                                                                         \
        DECLARE_IMREAD_ERROR_SUBTYPE_INNARDS(TypeName, BaseTypeName, DefaultMsg,        \
            static_strlen(ST(DefaultMsg)));                                             \
    };
#endif /// DECLARE_IMREAD_ERROR_SUBTYPE


DECLARE_IMREAD_ERROR_TYPE(CannotReadError,          "Read Error");
DECLARE_IMREAD_ERROR_TYPE(CannotWriteError,         "Write Error");
DECLARE_IMREAD_ERROR_TYPE(NotImplementedError,      "Not Implemented");
DECLARE_IMREAD_ERROR_TYPE(ProgrammingError,         "Programming Error");
DECLARE_IMREAD_ERROR_TYPE(OptionsError,             "Options Error");
DECLARE_IMREAD_ERROR_TYPE(WriteOptionsError,        "Write Options Error");
DECLARE_IMREAD_ERROR_TYPE(FileSystemError,          "File System Error");
DECLARE_IMREAD_ERROR_TYPE(FormatNotFound,           "File Format Not Found");

DECLARE_IMREAD_ERROR_SUBTYPE(JSONParseError,
                             std::runtime_error,    "JSON parsing error");
DECLARE_IMREAD_ERROR_SUBTYPE(JSONLogicError,
                             std::logic_error,      "JSON operator logic error");

DECLARE_IMREAD_ERROR_TYPE(JSONUseError,             "JSON library internal error");
DECLARE_IMREAD_ERROR_TYPE(JSONInvalidSchema,        "JSON schema parsing error");
DECLARE_IMREAD_ERROR_TYPE(JSONOutOfRange,           "JSON index value out of range");
DECLARE_IMREAD_ERROR_TYPE(JSONBadCast,              "Error casting JSON value");
DECLARE_IMREAD_ERROR_TYPE(JSONIOError,              "Error in JSON I/O");

DECLARE_IMREAD_ERROR_TYPE(HDF5IOError,              "Error in HDF5 I/O");
DECLARE_IMREAD_ERROR_TYPE(PNGIOError,               "Error in PNG/libpng I/O");
DECLARE_IMREAD_ERROR_TYPE(TIFFIOError,              "Error in TIFF/libtiff I/O");


}

#endif // LIBIMREAD_ERRORS_HH_
