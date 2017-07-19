/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_LIBIMREAD_CONFIG_LIBIMREAD_CONFIG_H_
#define APPS_LIBIMREAD_CONFIG_LIBIMREAD_CONFIG_H_

#include <string>
#include <vector>
#include <libimread/libimread.hpp>

#include "detail.hh"

namespace im {
    
    namespace config {
        
        const std::string version(IM_VERSION);
        
        const std::string prefix(IM_INSTALL_PREFIX);
        const std::string exec_prefix(IM_INSTALL_PREFIX);
        const std::string includes = detail::get_includes(IM_INCLUDE_DIRECTORIES);
        const std::string libs = detail::get_libs(IM_LINK_LIBRARIES);
        
        const std::string cflags = std::string(IM_COMPILE_OPTIONS) + " "
                                 + std::string(IM_COMPILE_FLAGS) + " "
                                 + includes;
        
        const std::string ldflags = std::string(IM_LINK_FLAGS) + " " + libs;
        
    } /* namespace config */
    
} /* namespace im */

#endif /// APPS_LIBIMREAD_CONFIG_LIBIMREAD_CONFIG_H_