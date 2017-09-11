
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/store.hh>
#include <libimread/serialization.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using store::stringmap;
    
    TEST_CASE("[serialization] Test store::detail::*_dumps() and store::detail::*_impl()",
              "[serialization-test-store-detail-dumps-store-detail-impl]")
    {
        stringmap source;
        
        source.set("yo",            "dogg");
        source.set("i_heard",       "you like");
        source.set("serialization", "and all of its methods");
        source.set("so",            "we put some");
        source.set("strings",       "in your strings so you can");
        source.set("encode",        "while you decode and");
        source.set("decode",        "while you encode");
        source.set("etc",           "and soforth");
        
        #define DEFINE_SERIALIZATION_FORMAT_SECTION(__name__)                                                                           \
                                                                                                                                        \
            SECTION("[serialization] » Testing store::detail::" # __name__ "_dumps() and store::detail::" # __name__ "_impl()")         \
            {                                                                                                                           \
                stringmap destination;                                                                                                  \
                std::string dump = store::detail::__name__##_dumps(source.mapping());                                                   \
                store::detail::__name__##_impl(dump, &destination);                                                                     \
                CHECK(source.to_string() == destination.to_string());                                                                   \
            }
        
        DEFINE_SERIALIZATION_FORMAT_SECTION(ini);
        
        // SECTION("[serialization] » Testing store::detail::ini_dumps() and store::detail::ini_impl()")
        // {
        //     stringmap destination;
        //     std::string dump = store::detail::ini_dumps(source.mapping());
        //     store::detail::ini_impl(dump, &destination);
        //     CHECK(source.to_string() == destination.to_string());
        // }
        
    }
    
} /// namespace (anon.)