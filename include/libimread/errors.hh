/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_ERRORS_HH_
#define LIBIMREAD_ERRORS_HH_

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>

#include <libimread/libimread.hpp>
#include <libimread/traits.hh>
#include <libimread/stringnormatives.hh>
#include <libimread/ansicolor.hh>

namespace im {
    
    template <typename ...Args> inline
    void __attribute__((nonnull (1, 3)))
         srsly(char const* title, ansi::ANSI const& color,
               char const* file, int line, Args&& ...args) {
        std::cerr  << color << im::stringify(title) << ansi::reset
          << "[ "  << ansi::yellow << im::stringify(file) << ansi::reset
          << " : " << ansi::red << im::stringify(line) << ansi::reset
          << " ]"  << std::endl
                   << im::stringmerge(std::forward<Args>(args)...)
                   << std::endl;
    }
    
    template <typename ...Args> inline
    std::string __attribute__((nonnull (1, 3)))
         emerg(char const* title, ansi::ANSI const& color,
               char const* file, int line, Args&& ...args) {
        std::ostringstream errstream;
        errstream  << color << im::stringify(title) << ansi::reset
          << "[ "  << ansi::yellow << im::stringify(file) << ansi::reset
          << " : " << ansi::red << im::stringify(line) << ansi::reset
          << " ]"  << std::endl
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
                FF("    In function: %s %s %s\n",                                       \
                    ansi::yellow.c_str(), __PRETTY_FUNCTION__, ansi::reset.c_str()),    \
                __VA_ARGS__);                                                           \
        std::exit(-1);                                                                  \
    }
#endif /// imread_assert

#ifndef imread_raise_default
#define imread_raise_default(Exception)                                                 \
    throw im::Exception(im::emerg("\n[ ERROR > " ST(Exception) " ]\n",                  \
        ansi::lightred, __FILE__, __LINE__,                                             \
            FF("    In function: %s %s %s\n",                                           \
                ansi::red.c_str(), __PRETTY_FUNCTION__, ansi::reset.c_str()),           \
            FF("\t%s", im::Exception::default_message)));
#endif /// imread_raise_default

#ifndef imread_raise
#define imread_raise(Exception, ...)                                                    \
    throw im::Exception(im::emerg("\n[ ERROR > " ST(Exception) " ]\n",                  \
        ansi::lightred, __FILE__, __LINE__,                                             \
            FF("    In function: %s %s %s\n",                                           \
                ansi::red.c_str(), __PRETTY_FUNCTION__, ansi::reset.c_str()),           \
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
DECLARE_IMREAD_ERROR_TYPE(MetadataReadError,        "Metadata Read Error");
DECLARE_IMREAD_ERROR_TYPE(MetadataWriteError,       "Metadata Write Error");
DECLARE_IMREAD_ERROR_TYPE(NotImplementedError,      "Not Implemented");
DECLARE_IMREAD_ERROR_TYPE(ProgrammingError,         "Programming Error");
DECLARE_IMREAD_ERROR_TYPE(OptionsError,             "Options Error");
DECLARE_IMREAD_ERROR_TYPE(WriteOptionsError,        "Write Options Error");
DECLARE_IMREAD_ERROR_TYPE(FileSystemError,          "File System Error");
DECLARE_IMREAD_ERROR_TYPE(GZipIOError,              "GZip I/O Error");
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
DECLARE_IMREAD_ERROR_TYPE(PListIOError,             "Error in Property List I/O");
DECLARE_IMREAD_ERROR_TYPE(IniIOError,               "Error in Ini File I/O");
DECLARE_IMREAD_ERROR_TYPE(YAMLIOError,              "Error in YAML I/O");

DECLARE_IMREAD_ERROR_TYPE(HDF5IOError,              "Error in HDF5 I/O");
DECLARE_IMREAD_ERROR_TYPE(JPEGIOError,              "Error in JPEG/jpeglib I/O");
DECLARE_IMREAD_ERROR_TYPE(PNGIOError,               "Error in PNG/libpng I/O");
DECLARE_IMREAD_ERROR_TYPE(PPMIOError,               "Error in PPM binary I/O");
DECLARE_IMREAD_ERROR_TYPE(TIFFIOError,              "Error in TIFF/libtiff I/O");

}

#endif /// LIBIMREAD_ERRORS_HH_
