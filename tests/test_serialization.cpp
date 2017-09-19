
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/store.hh>
#include <libimread/serialization.hh>

#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>

#ifdef __APPLE__
#include <libimread/corefoundation.hh>
#endif

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
        source.set("encode",        "while you decode, and");
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
        DEFINE_SERIALIZATION_FORMAT_SECTION(urlparam);
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
        source.set("encode",        "while you decode, and");
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
                    CHECK(store::detail::dump(dump, tn.pathname.str()));                                                                    \
                    tf.open();                                                                                                              \
                    tf.stream << dump;                                                                                                      \
                    tf.close();                                                                                                             \
                    WTF("SERIALIZATION - " # __name__ " dump:", "\n" + asterisks + "\n" + dump + "\n" + asterisks);                         \
                    WTF("SERIALIZATION - " # __name__ " file:", tn.pathname.str());                                                         \
                    std::string manual_load = store::detail::load(tn.pathname.str());                                                       \
                    std::string auto_load = store::detail::load(tf.filepath.str());                                                         \
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
                    CHECK(store::detail::dump(dump, tn.pathname.str()));                                                                    \
                    tf.open();                                                                                                              \
                    tf.stream << dump;                                                                                                      \
                    tf.close();                                                                                                             \
                    std::string manual_load = store::detail::load(tn.pathname.str());                                                       \
                    std::string auto_load = store::detail::load(tf.filepath.str());                                                         \
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
        DEFINE_SERIALIZATION_FORMAT_SECTION(urlparam);
        DEFINE_SERIALIZATION_FORMAT_SECTION(yaml);
    }
    
    #undef DEFINE_SERIALIZATION_FORMAT_SECTION
    
    #ifdef __APPLE__
    #define STRINGNULL() store::detail::value_for_null<std::string>()
    
    TEST_CASE("[serialization] Inspect static “im::detail::master” stringmap",
              "[serialization-inspect-static-im-detail-master-stringmap]")
    {
        
        store::stringmap map;
        im::detail::initialize_stringmap(map);
        
        REQUIRE(map.count() > 0);
        
        CHECK(map.get("CFBundleDevelopmentRegion") != STRINGNULL());
        CHECK(map.get("CFBundleExecutable") != STRINGNULL());
        CHECK(map.get("CFBundleIdentifier") != STRINGNULL());
        CHECK(map.get("CFBundleInfoDictionaryVersion") != STRINGNULL());
        CHECK(map.get("CFBundleName") != STRINGNULL());
        CHECK(map.get("CFBundlePackageType") != STRINGNULL());
        CHECK(map.get("CFBundleShortVersionString") != STRINGNULL());
        CHECK(map.get("CFBundleVersion") != STRINGNULL());
        CHECK(map.get("LSMinimumSystemVersion") != STRINGNULL());
        CHECK(map.get("LSRequiresIPhoneOS") != STRINGNULL());
        CHECK(map.get("NSHumanReadableCopyright") != STRINGNULL());
        CHECK(map.get("NSMainStoryboardFile") != STRINGNULL());
        CHECK(map.get("NSPrincipalClass") != STRINGNULL());
        CHECK(map.get("UILaunchStoryboardName") != STRINGNULL());
        CHECK(map.get("UIMainStoryboardFile") != STRINGNULL());
        
    }
    #endif
    
} /// namespace (anon.)