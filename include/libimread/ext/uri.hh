/// Copyright 2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_URI_HH_
#define LIBIMREAD_EXT_URI_HH_

#include <string>

namespace im {
    
    namespace uri {
        
        std::string encode(std::string const&);
        std::string decode(std::string const&);
        
    } /// namespace uri
    
} /// namespace im

#endif /// LIBIMREAD_EXT_URI_HH_