
#include <cstdlib>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>

#include <libimread/store.hh>
#include <libimread/env.hh>

// #define COLLECT_TEMPORARIES 1
#include "helpers/collect.hh"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    
    TEST_CASE("[environment] List the environment variables of the current process",
              "[environment-list-environment-variables-current-process]")
    {
        store::env viron;
        
        {
            NamedTemporaryFile tf(".json");
            tf.open();
            tf.stream << viron.mapping_json()
                      << std::endl;
            CHECK(tf.close());
            CHECK(COLLECT(tf.filepath));
        }
        
        REQUIRE(viron.count() > 0);
        int oldcount = viron.count();
        // WTF("Total environment variables: ", viron.count());
        
        for (std::string const& name : viron.list()) {
            // WTF("Environment variable: ", name);
            if (std::strcmp(std::getenv(name.c_str()), "") != 0) {
                CHECK(viron.get(name) == std::getenv(name.c_str()));
            }
        }
        
        std::string nk("YO_DOGG");
        std::string nv("I heard you like environment variables");
        
        CHECK(std::getenv(nk.c_str()) == nullptr);
        CHECK(viron.get(nk) == viron.null_value());
        
        REQUIRE(viron.set(nk, nv));
        CHECK(std::getenv(nk.c_str()) == std::string("I heard you like environment variables"));
        CHECK(viron.get(nk) == "I heard you like environment variables");
        CHECK(viron.count() == oldcount + 1);
        
        REQUIRE(viron.del(nk));
        CHECK(std::getenv(nk.c_str()) == nullptr);
        CHECK(viron.get(nk) == viron.null_value());
        CHECK(viron.count() == oldcount);
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
            if (std::strcmp(std::getenv(name.c_str()), "") != 0) {
                CHECK(viron.get(name) == std::getenv(name.c_str()));
            }
        }
        
        std::string nk("YO_DOGG");
        std::string nv("I heard you like environment variables");
        REQUIRE(viron.set(nk, nv));
        
        store::stringmap memcopy2(viron);
        REQUIRE(viron.count() == memcopy2.count());
        
        /// WAIT WHAT THE FUCK, THIS IS NOT SUPPOSED TO HAPPEN
        // CHECK(viron.get(nk) == memcopy.get(nk));
        
        CHECK(viron.count() == memcopy2.count());
        CHECK(viron.count() == memcopycount + 1);
        CHECK(viron.count() == memcopy.count() + 1);
        CHECK(viron.get(nk) == memcopy2.get(nk));
        
        for (std::string const& name : memcopy2.list()) {
            if (std::strcmp(std::getenv(name.c_str()), "") != 0) {
                CHECK(viron.get(name) == std::getenv(name.c_str()));
            }
        }
        
        REQUIRE(viron.del(nk));
        CHECK(viron.get(nk) == viron.null_value());
        CHECK(viron.count() == memcopy.count());
        CHECK(viron.count() == memcopycount);
        // CHECK(viron.get(nk) == memcopy.get(nk));
        CHECK(viron.count() == memcopy2.count() - 1);
        
        {
            NamedTemporaryFile vt(".json");
            vt.open();
            vt.stream << viron.mapping_json()
                      << std::endl;
            CHECK(vt.close());
            CHECK(COLLECT(vt.filepath));
        }
        
        {
            NamedTemporaryFile mt(".json");
            mt.open();
            mt.stream << memcopy.mapping_json()
                      << std::endl;
            CHECK(mt.close());
            CHECK(COLLECT(mt.filepath));
        }
        
        bool compare_eq = (viron == memcopy);
        
        REQUIRE(compare_eq);
        
        // path vironpth = td.dirpath.join("viron-dump.json");
        // path memorypth = td.dirpath.join("memory-dump.json");
        
    }
    
} /// namespace (anon.)