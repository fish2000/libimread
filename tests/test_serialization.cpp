
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/store.hh>
#include <libimread/serialization.hh>

#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>

#include "include/test_data.hpp"
#include "include/catch.hpp"

#define SERIALIZATION_PRINT_DUMPS 0

namespace {
    
    using store::stringmap;
    using filesystem::path;
    using filesystem::TemporaryName;
    using filesystem::NamedTemporaryFile;
    
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
                    CHECK(source == destination);                                                                                           \
                }
        
        #else
            #define DEFINE_SERIALIZATION_FORMAT_SECTION(__name__)                                                                           \
                                                                                                                                            \
                SECTION("[serialization] » Testing store::detail::" # __name__ "_dumps() and store::detail::" # __name__ "_impl()")         \
                {                                                                                                                           \
                    stringmap destination;                                                                                                  \
                    std::string dump = store::detail::__name__##_dumps(source.mapping());                                                   \
                    store::detail::__name__##_impl(dump, &destination);                                                                     \
                    CHECK(source == destination);                                                                                           \
                }
        
        #endif
        
        DEFINE_SERIALIZATION_FORMAT_SECTION(ini);
        DEFINE_SERIALIZATION_FORMAT_SECTION(json);
        DEFINE_SERIALIZATION_FORMAT_SECTION(plist);
        DEFINE_SERIALIZATION_FORMAT_SECTION(yaml);
    }
    
    #undef DEFINE_SERIALIZATION_FORMAT_SECTION
    
    TEST_CASE("[serialization] Test string I/O helper functions",
              "[serialization-test-string-io-helper-functions]")
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
                    TemporaryName tn("." # __name__);                                                                                       \
                    NamedTemporaryFile tf("." # __name__);                                                                                  \
                    REQUIRE(tn.pathname != tf.filepath);                                                                                    \
                    std::string dump = store::detail::__name__##_dumps(source.mapping());                                                   \
                    CHECK(store::detail::string_dump(dump, tn.pathname.str()));                                                             \
                    tf.open();                                                                                                              \
                    tf.stream << dump;                                                                                                      \
                    tf.close();                                                                                                             \
                    WTF("SERIALIZATION - " # __name__ " dump:", "\n" + asterisks + "\n" + dump + "\n" + asterisks);                         \
                    WTF("SERIALIZATION - " # __name__ " file:", tn.pathname.str());                                                         \
                    std::string manual_load = store::detail::string_load(tn.pathname.str());                                                \
                    std::string auto_load = store::detail::string_load(tf.filepath.str());                                                  \
                    stringmap destination(auto_load, store::detail::for_path(tn.pathname.str()));                                           \
                    CHECK(manual_load == auto_load);                                                                                        \
                    CHECK(manual_load == dump);                                                                                             \
                    CHECK(auto_load == dump);                                                                                               \
                    CHECK(source == destination);                                                                                           \
                }
        
        #else
            #define DEFINE_SERIALIZATION_FORMAT_SECTION(__name__)                                                                           \
                                                                                                                                            \
                SECTION("[serialization] » Testing store::detail::" # __name__ "_dumps() and store::detail::" # __name__ "_impl()")         \
                {                                                                                                                           \
                    TemporaryName tn("." # __name__);                                                                                       \
                    NamedTemporaryFile tf("." # __name__);                                                                                  \
                    REQUIRE(tn.pathname != tf.filepath);                                                                                    \
                    std::string dump = store::detail::__name__##_dumps(source.mapping());                                                   \
                    CHECK(store::detail::string_dump(dump, tn.pathname.str()));                                                             \
                    tf.open();                                                                                                              \
                    tf.stream << dump;                                                                                                      \
                    tf.close();                                                                                                             \
                    std::string manual_load = store::detail::string_load(tn.pathname.str());                                                \
                    std::string auto_load = store::detail::string_load(tf.filepath.str());                                                  \
                    stringmap destination(auto_load, store::detail::for_path(tn.pathname.str()));                                           \
                    CHECK(manual_load == auto_load);                                                                                        \
                    CHECK(manual_load == dump);                                                                                             \
                    CHECK(auto_load == dump);                                                                                               \
                    CHECK(source == destination);                                                                                           \
                }
        
        #endif
        
        DEFINE_SERIALIZATION_FORMAT_SECTION(ini);
        DEFINE_SERIALIZATION_FORMAT_SECTION(json);
        DEFINE_SERIALIZATION_FORMAT_SECTION(plist);
        DEFINE_SERIALIZATION_FORMAT_SECTION(yaml);
    }
    
    #undef DEFINE_SERIALIZATION_FORMAT_SECTION
    
} /// namespace (anon.)