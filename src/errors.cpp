/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/ext/errors/terminator.hh>
#include <libimread/errors.hh>
#include <libimread/objc-rt.hh>

namespace im {

#ifndef ST
#define ST(s) "" #s 
#endif /// ST

#ifndef DECLARE_IMREAD_ERROR_DEFAULT
#define DECLARE_IMREAD_ERROR_DEFAULT(TypeName, DefaultMsg) \
    constexpr char TypeName::default_message[static_strlen(ST(DefaultMsg))];
#endif /// DECLARE_IMREAD_ERROR_DEFAULT

    DECLARE_IMREAD_ERROR_DEFAULT(CannotReadError,          "Read Error");
    DECLARE_IMREAD_ERROR_DEFAULT(CannotWriteError,         "Write Error");
    DECLARE_IMREAD_ERROR_DEFAULT(NotImplementedError,      "Not Implemented");
    DECLARE_IMREAD_ERROR_DEFAULT(ProgrammingError,         "Programming Error");
    DECLARE_IMREAD_ERROR_DEFAULT(OptionsError,             "Options Error");
    DECLARE_IMREAD_ERROR_DEFAULT(WriteOptionsError,        "Write Options Error");
    DECLARE_IMREAD_ERROR_DEFAULT(FileSystemError,          "File System Error");
    DECLARE_IMREAD_ERROR_DEFAULT(FormatNotFound,           "File Format Not Found");
    
    DECLARE_IMREAD_ERROR_DEFAULT(JSONParseError,           "JSON parsing error");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONLogicError,           "JSON operator logic error");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONUseError,             "JSON library internal error");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONInvalidSchema,        "JSON schema parsing error");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONOutOfRange,           "JSON index value out of range");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONBadCast,              "Error casting JSON value");
    DECLARE_IMREAD_ERROR_DEFAULT(HDF5IOError,              "Error in HDF5 I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(HaltWalking,              "Halt Walking");

}