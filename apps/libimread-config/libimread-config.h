/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_LIBIMREAD_CONFIG_H_
#define APPS_LIBIMREAD_CONFIG_H_

#include <libimread/libimread.hpp>
#include <libimread/ext/pystring.hh>

namespace im {
    
    namespace config {
        
        const std::string prefix = std::string(IM_INSTALL_PREFIX);
        const std::string exec_prefix = prefix;
        const std::string includes = std::string("-I") + pystring::replace(IM_INCLUDE_DIRECTORIES, ";", " -I");
        const std::string libs = std::string(IM_LINK_LIBRARIES);
        const std::string cflags = includes + " " + std::string(IM_COMPILE_OPTIONS);
        const std::string ldflags = std::string(IM_LINK_FLAGS) + " " + libs;
        
        const std::string version = IM_VERSION;
        
    };
    
};


#endif /// APPS_LIBIMREAD_CONFIG_H_