/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/objc-rt/traits.hh>

namespace objc {
    
    namespace traits {
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wnullability-completeness"
        #define DECLARE_NULL_SPECIFIER_TRAITS(name, signifier)                          \
            const name##_ptr<> name##_cast = name##_ptr<>();
        
        DECLARE_NULL_SPECIFIER_TRAITS(nullable,      _Nullable);
        DECLARE_NULL_SPECIFIER_TRAITS(nonnull,       _Nonnull);
        DECLARE_NULL_SPECIFIER_TRAITS(unspecified,   _Null_unspecified);
        #pragma clang diagnostic pop
        
        
    }
    
}
