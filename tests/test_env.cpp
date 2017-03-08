
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
// #include <libimread/store.hh>
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
        WTF("Total environment variables: ", viron.count());
        
        for (std::string const& name : viron.list()) {
            WTF("Environment variable: ", name);
        }
    }
    
} /// namespace (anon.)