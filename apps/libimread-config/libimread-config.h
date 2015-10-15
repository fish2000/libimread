/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_LIBIMREAD_CONFIG_H_
#define APPS_LIBIMREAD_CONFIG_H_

#include <libimread/libimread.hpp>

namespace im {
    
    namespace config {
        
        const std::string prefix = IM_INSTALL_PREFIX;
        const std::string exec_prefix = prefix;
        const std::string includes = IM_INCLUDE_DIRECTORIES;
        const std::string libs = IM_LINK_LIBRARIES;
        const std::string cflags = includes + " " + IM_COMPILE_OPTIONS;
        const std::string ldflags = std::string(IM_LINK_FLAGS) + " " + libs;
        
        const std::string version = IM_VERSION;
        
    };
    
};


#endif /// APPS_LIBIMREAD_CONFIG_H_