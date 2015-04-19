/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LPC_FORMATS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_FORMATS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <memory>
#include <string>
#include <libimread/libimread.hpp>
#include <libimread/imageformat.hh>

namespace im {
    
    std::unique_ptr<ImageFormat> get_format(const char*);
    std::unique_ptr<ImageFormat> for_filename(const char*);
    std::unique_ptr<ImageFormat> for_filename(std::string&);
    std::unique_ptr<ImageFormat> for_filename(const std::string&);
    const char *magic_format(byte_source*);

}

#endif // LPC_FORMATS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
