/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/ext/errors/terminator.hh>
#include <libimread/errors.hh>

/// set up the terminator
#ifdef IM_TERMINATOR
static bool did_setup_terminator = terminator::setup();
#endif

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
    DECLARE_IMREAD_ERROR_DEFAULT(MetadataReadError,        "Metadata Read Error");
    DECLARE_IMREAD_ERROR_DEFAULT(MetadataWriteError,       "Metadata Write Error");
    DECLARE_IMREAD_ERROR_DEFAULT(NotImplementedError,      "Not Implemented");
    DECLARE_IMREAD_ERROR_DEFAULT(ProgrammingError,         "Programming Error");
    DECLARE_IMREAD_ERROR_DEFAULT(OptionsError,             "Options Error");
    DECLARE_IMREAD_ERROR_DEFAULT(WriteOptionsError,        "Write Options Error");
    DECLARE_IMREAD_ERROR_DEFAULT(FileSystemError,          "File System Error");
    DECLARE_IMREAD_ERROR_DEFAULT(GZipIOError,              "GZip I/O Error");
    DECLARE_IMREAD_ERROR_DEFAULT(FormatNotFound,           "File Format Not Found");
    
    DECLARE_IMREAD_ERROR_DEFAULT(JSONParseError,           "JSON parsing error");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONLogicError,           "JSON operator logic error");
    
    DECLARE_IMREAD_ERROR_DEFAULT(JSONUseError,             "JSON library internal error");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONInvalidSchema,        "JSON schema parsing error");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONOutOfRange,           "JSON index value out of range");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONBadCast,              "Error casting JSON value");
    DECLARE_IMREAD_ERROR_DEFAULT(JSONIOError,              "Error in JSON I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(PListIOError,             "Error in Property List I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(IniIOError,               "Error in Ini File I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(YAMLIOError,              "Error in YAML I/O");
    
    DECLARE_IMREAD_ERROR_DEFAULT(HDF5IOError,              "Error in HDF5 I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(JPEGIOError,              "Error in JPEG/jpeglib I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(PNGIOError,               "Error in PNG/libpng I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(PPMIOError,               "Error in PPM binary I/O");
    DECLARE_IMREAD_ERROR_DEFAULT(TIFFIOError,              "Error in TIFF/libtiff I/O");
}