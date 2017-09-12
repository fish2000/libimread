
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/store.hh>
#include <libimread/serialization.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

#define SERIALIZATION_PRINT_DUMPS 1

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
        
        #if(SERIALIZATION_PRINT_DUMPS)
        std::string asterisks(120, '*');
        
            #define DEFINE_SERIALIZATION_FORMAT_SECTION(__name__)                                                                           \
                                                                                                                                            \
                SECTION("[serialization] » Testing store::detail::" # __name__ "_dumps() and store::detail::" # __name__ "_impl()")         \
                {                                                                                                                           \
                    stringmap destination;                                                                                                  \
                    std::string dump = store::detail::__name__##_dumps(source.mapping());                                                   \
                    WTF("SERIALIZATION - " # __name__ " dump:", "\n" + asterisks + "\n" + dump + "\n" + asterisks);                         \
                    store::detail::__name__##_impl(dump, &destination);                                                                     \
                    CHECK(source.to_string() == destination.to_string());                                                                   \
                }
        
        #else
            #define DEFINE_SERIALIZATION_FORMAT_SECTION(__name__)                                                                           \
                                                                                                                                            \
                SECTION("[serialization] » Testing store::detail::" # __name__ "_dumps() and store::detail::" # __name__ "_impl()")         \
                {                                                                                                                           \
                    stringmap destination;                                                                                                  \
                    std::string dump = store::detail::__name__##_dumps(source.mapping());                                                   \
                    store::detail::__name__##_impl(dump, &destination);                                                                     \
                    CHECK(source.to_string() == destination.to_string());                                                                   \
                }
        
        #endif
        
        DEFINE_SERIALIZATION_FORMAT_SECTION(ini);
        DEFINE_SERIALIZATION_FORMAT_SECTION(json);
        DEFINE_SERIALIZATION_FORMAT_SECTION(plist);
        DEFINE_SERIALIZATION_FORMAT_SECTION(yaml);
    }
    
} /// namespace (anon.)