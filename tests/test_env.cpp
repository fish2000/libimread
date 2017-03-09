
#include <cstdlib>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
// #include <libimread/ext/filesystem/mode.h>
// #include <libimread/ext/filesystem/path.h>
// #include <libimread/ext/filesystem/attributes.h>
// #include <libimread/ext/filesystem/directory.h>
// #include <libimread/ext/filesystem/temporary.h>
// #include <libimread/ext/filesystem/nowait.h>
// #include <libimread/ext/JSON/json11.h>

// #include <libimread/file.hh>
// #include <libimread/filehandle.hh>
#include <libimread/store.hh>
// #include <libimread/rocks.hh>
#include <libimread/env.hh>

#include "include/catch.hpp"

namespace {
    
    // using filesystem::path;
    // using filesystem::switchdir;
    // using filesystem::resolver;
    // using filesystem::NamedTemporaryFile;
    // using filesystem::TemporaryDirectory;
    
    // using filesystem::detail::nowait_t;
    // using filesystem::detail::stringvec_t;
    // using filesystem::attribute::accessor_t;
    // using filesystem::attribute::detail::nullstring;
    
    // using im::FileSource;
    // using im::FileSink;
    
    TEST_CASE("[environment] List the environment variables of the current process",
              "[environment-list-environment-variables-current-process]")
    {
        store::env viron;
        
        REQUIRE(viron.count() > 0);
        int oldcount = viron.count();
        // WTF("Total environment variables: ", viron.count());
        
        for (std::string const& name : viron.list()) {
            // WTF("Environment variable: ", name);
            CHECK(viron.get(name) == std::getenv(name.c_str()));
        }
        
        std::string nk("YO_DOGG");
        std::string nv("I heard you like environment variables");
        
        CHECK(std::getenv(nk.c_str()) == nullptr);
        CHECK(viron.get(nk) == viron.null_value());
        viron.set(nk, nv);
        CHECK(std::getenv(nk.c_str()) == std::string("I heard you like environment variables"));
        CHECK(viron.get(nk) == "I heard you like environment variables");
        CHECK(viron.count() == oldcount + 1);
    }
    
    
    TEST_CASE("[environment] Copy/Dump/Load environment variables via the store API",
              "[environment-copy-dump-load-environment-variables-store-API]")
    {
        store::env viron;
        int vironcount = viron.count();
        
        store::stringmap memcopy(viron);
        int memcopycount = memcopy.count();
        
        REQUIRE(vironcount == memcopycount);
        
        for (std::string const& name : memcopy.list()) {
            CHECK(viron.get(name) == std::getenv(name.c_str()));
        }
        
        std::string nk("YO_DOGG");
        std::string nv("I heard you like environment variables");
        viron.set(nk, nv);
        
        store::stringmap memcopy2(viron);
        REQUIRE(viron.count() == memcopy2.count());
        
        /// WAIT WHAT THE FUCK, THIS IS NOT SUPPOSED TO HAPPEN
        // REQUIRE(viron.count() == memcopycount + 1);
        CHECK(viron.get(nk) == memcopy2.get(nk));
        CHECK(viron.get(nk) == memcopy.get(nk));
    }
    
} /// namespace (anon.)