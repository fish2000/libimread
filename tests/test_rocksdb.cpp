
#include <iostream>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/attributes.h>
#include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/filesystem/nowait.h>
#include <libimread/ext/JSON/json11.h>

#include <libimread/file.hh>
#include <libimread/filehandle.hh>
#include <libimread/store.hh>
#include <libimread/rocks.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::switchdir;
    // using filesystem::resolver;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    
    using filesystem::detail::nowait_t;
    using filesystem::detail::stringvec_t;
    using filesystem::attribute::accessor_t;
    using filesystem::attribute::detail::nullstring;
    
    using im::FileSource;
    using im::FileSink;
    
    TEST_CASE("[rocksdb] Create, update, read from, and ultimately destroy a RocksDB database",
              "[rocksdb-create-update-readfrom-ultimately-destroy-database]")
    {
        /// set up the Rocks instance
        TemporaryDirectory td("test-rocks");
        path rockspth = td.dirpath.join("data.db");
        store::rocks database(rockspth.str());
        
        /// load some test data
        path basedir = path(im::test::basedir);
        path jsondir = basedir / "json";
        path jsondoc = jsondir / "assorted.json";
        
        /// keep a manual count
        std::size_t manualcount = 0;
        
        /// list of dicts, each with an _id field
        Json dictlist = Json::load(jsondoc.str());
        int max = dictlist.size();
        for (int idx = 0; idx < max; ++idx) {
            Json dict = dictlist.at(idx);
            std::string prefix(dict.get("_id"));
            for (std::string const& key : dict.keys()) {
                /// construct a prefixed key for use in Rocks
                std::string nk = prefix + ":" + key;
                WTF("KEY: ", nk, "VALUE: ", std::string(dict.get(key)));
                std::string nv = dict.cast<std::string>(key);
                database.set(nk, nv);
                /// increment manual count
                ++manualcount;
            }
        }
        
        REQUIRE(database.count() == manualcount);
        
        for (std::string const& key : database.list()) {
            WTF("KEY: ", key);
            CHECK(database.get(key) != database.null_value());
        }
        
    }

} /// namespace (anon.)
