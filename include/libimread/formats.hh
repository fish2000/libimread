/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_FORMATS_HH_
#define LIBIMREAD_FORMATS_HH_

#include <memory>
#include <string>
#include <cstring>
#include <libimread/libimread.hpp>
#include <libimread/imageformat.hh>

namespace im {
    
    namespace detail {
        inline const char *split_filename(const char *const filename,
                                          char *const body = 0) {
            if (!filename) {
                if (body) { *body = 0; }
                return 0;
            }
        
            const char *p = 0;
            for (const char *np = filename; np >= filename && (p = np);
                 np = std::strchr(np, '.') + 1) {
            }
            if (p == filename) {
                if (body) { std::strcpy(body, filename); }
                return filename + std::strlen(filename);
            }
        
            const unsigned int l = static_cast<unsigned int>(p - filename - 1);
            if (body) {
                std::memcpy(body, filename, l);
                body[l] = 0;
            }
        
            return p;
        }
        
        inline bool ext(const char *format, const char *suffix) {
            return !std::strcmp(format, suffix);
        }
    }
    
    std::unique_ptr<ImageFormat> get_format(const char*);
    std::unique_ptr<ImageFormat> for_filename(const char*);
    std::unique_ptr<ImageFormat> for_filename(std::string&);
    std::unique_ptr<ImageFormat> for_filename(const std::string&);
    const char *magic_format(byte_source*);

}

#endif /// LIBIMREAD_FORMATS_HH_
