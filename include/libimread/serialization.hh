/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_SERIALIZATION_HH_
#define LIBIMREAD_INCLUDE_SERIALIZATION_HH_

#include <string>
#include <libimread/libimread.hpp>
#include <libimread/store.hh>

namespace store {
    
    namespace detail {
        
        std::string json_dumps(store::stringmapper::stringmap_t const&);
        void json_impl(std::string const&, store::stringmapper*);
        
        std::string plist_dumps(store::stringmapper::stringmap_t const&);
        void plist_impl(std::string const&, store::stringmapper*);
        
        std::string yaml_dumps(store::stringmapper::stringmap_t const&);
        void yaml_impl(std::string const&, store::stringmapper*);
        
        store::stringmapper::formatter for_path(std::string const&);
        std::string string_load(std::string const&);
        bool string_dump(std::string const&, std::string const&, bool overwrite = false);
    }
    
}

#endif /// LIBIMREAD_INCLUDE_SERIALIZATION_HH_