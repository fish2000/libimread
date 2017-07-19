/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_LIBIMREAD_DETAIL_HH_
#define APPS_LIBIMREAD_DETAIL_HH_

#include <string>
#include <vector>
#include <libimread/libimread.hpp>

namespace im {
    
    namespace config {
        
        namespace detail {
            
            std::string get_includes(std::string const& inclist);
            std::string get_libs(std::string const& liblist);
            
        } /* namespace detail */
        
    } /* namespace config */
    
} /* namespace im */

#endif /// APPS_LIBIMREAD_DETAIL_HH_