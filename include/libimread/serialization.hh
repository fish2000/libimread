/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_SERIALIZATION_HH_
#define LIBIMREAD_INCLUDE_SERIALIZATION_HH_

#include <string>
#include <plist/plist++.h>

#include <libimread/libimread.hpp>
#include <libimread/store.hh>
#include <libimread/ext/JSON/json11.h>

namespace store {
    
    namespace detail {
        
        bool json_dump(store::stringmapper::stringmap_t const&, std::string const&, bool);
        void json_map_impl(Json const&, store::stringmapper*);
        void json_impl(Json const&, store::stringmapper*);
        
        bool plist_dump(store::stringmapper::stringmap_t const&, std::string const&, bool overwrite = false);
        PList::Dictionary plist_load(std::string const&);
        void plist_impl(PList::Dictionary&, store::stringmapper*);
    }
    
}

#endif /// LIBIMREAD_INCLUDE_SERIALIZATION_HH_