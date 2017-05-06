
#include <iostream>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
// #include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>
// #include <libimread/ext/filesystem/attributes.h>
// #include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/temporary.h>
// #include <libimread/ext/filesystem/nowait.h>
#include <libimread/ext/JSON/json11.h>
#include <libimread/ext/pystring.hh>

#include <libimread/file.hh>
#include <libimread/filehandle.hh>
#include <libimread/store.hh>
#include <libimread/rocks.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    // using filesystem::switchdir;
    // using filesystem::resolver;
    using filesystem::TemporaryName;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    
    // using filesystem::detail::nowait_t;
    using filesystem::detail::stringvec_t;
    // using filesystem::attribute::accessor_t;
    // using filesystem::attribute::detail::nullstring;
    
    using im::FileSource;
    using im::FileSink;
    
    TEST_CASE("[rocksdb] Create, update, read from, and ultimately destroy a RocksDB database",
              "[rocksdb-create-update-readfrom-ultimately-destroy-database]")
    {
        /// set up the Rocks instance
        TemporaryDirectory td("test-rocks");
        path rockspth = td.dirpath.join("data.db");
        path rocksjsonpth = td.dirpath.join("rocks-dump.json");
        path memoryjsonpth = td.dirpath.join("memory-dump.json");
        store::rocks database(rockspth.str());
        
        /// load some test data
        path basedir = path(im::test::basedir);
        path jsondir = basedir / "json";
        path jsondoc = jsondir / "assorted.json";
        
        /// keep a manual count
        std::size_t manualcount = 0;
        
        /// list of dicts, each with an _id field
        REQUIRE(jsondoc.is_readable());
        Json dictlist = Json::load(jsondoc.str());
        int max = dictlist.size();
        
        for (int idx = 0; idx < max; ++idx) {
            Json dict = dictlist.at(idx);
            std::string prefix(dict.get("_id"));
            
            for (std::string const& key : dict.keys()) {
                /// construct a prefixed key for use in Rocks
                std::string nk = prefix + ":" + key;
                
                // WTF("KEY: ", nk, "VALUE: ", std::string(dict.get(key)));
                std::string nv = dict.cast<std::string>(key);
                CHECK(database.set(nk, nv));
                
                /// increment manual count
                ++manualcount;
            }
        }
        
        REQUIRE(database.count() == manualcount);
        
        for (std::string const& key : database.list()) {
            CHECK(database.get(key) != database.null_value());
        }
        
        store::stringmap memcopy(database);
        
        REQUIRE(database.count() == memcopy.count());
        
        for (std::string const& key : database.list()) {
            CHECK(database.get(key) == memcopy.get(key));
            CHECK(memcopy.get(key) != memcopy.null_value());
        }
        
        // Json(database.mapping()).dump(rocksjsonpth.str());
        // Json(memcopy.mapping()).dump(memoryjsonpth.str());
        database.dump(rocksjsonpth.str());
        memcopy.dump(memoryjsonpth.str());
        
        REQUIRE(rocksjsonpth.exists());
        REQUIRE(rocksjsonpth.is_file());
        REQUIRE(memoryjsonpth.exists());
        REQUIRE(memoryjsonpth.is_file());
        
        SECTION("[rocksdb] » Reconstitute the Rocks database using store::stringmap::load()")
        {
            // WTF("PRE-CONSTITUTED FILE SIZE: ", rocksjsonpth.filesize());
            
            store::stringmap reconstituted = store::stringmap::load(rocksjsonpth.str());
            stringvec_t rekeyed = reconstituted.list();
            stringvec_t revalued;
            
            // WTF("RETURNED RECONSTITUTED: ",
            //     FF("%i mapped indexes",   reconstituted.count()),
            //     FF("%i string indexes\n", rekeyed.size()),
            //     pystring::join(", \n\t", rekeyed));
            
            revalued.reserve(rekeyed.size());
            std::for_each(rekeyed.begin(), rekeyed.end(), [&](std::string const& key) {
                std::string value(reconstituted.get(key));
                CHECK(value != reconstituted.null_value());
                if (value != reconstituted.null_value()) {
                    revalued.emplace_back(value.size() < 40 ? std::move(value) : value.substr(0, 40));
                }
            });
            
            // WTF("RETURNED RECONSTITUTED: ",
            //     FF("%i indexed values\n", revalued.size()),
            //     pystring::join(", \n\t", revalued));
            
            REQUIRE(rekeyed.size() == revalued.size());
            
            for (std::string const& key : reconstituted.list()) {
                if (key == reconstituted.null_key()) {
                    // WTF("NULL KEY FOUND");
                } else {
                    // WTF("KEY: ", key);
                    CHECK(database.get(key) == reconstituted.get(key));
                    CHECK(reconstituted.get(key) != reconstituted.null_value());
                }
            }
        }
        
        SECTION("[rocksdb] » Copy the Rocks database using value_copy() and xattr")
        {
            FileSource source(rocksjsonpth);
            store::value_copy(database, source);
            for (std::string const& key : database.list()) {
                CHECK(database.get(key) == source.get(key));
                CHECK(source.get(key) != source.null_value());
            }
        }
        
        SECTION("[rocksdb] » Copy the string database using value_copy() and xattr")
        {
            FileSource source(memoryjsonpth);
            store::value_copy(database, source);
            for (std::string const& key : database.list()) {
                CHECK(database.get(key) == source.get(key));
                CHECK(source.get(key) != source.null_value());
            }
        }
        
        SECTION("[rocksdb] » Copy the Rocks database using value_copy() and xattr with a NamedTemporaryFile")
        {
            NamedTemporaryFile tf(".json");
            REQUIRE(tf.filepath.remove());
            FileSink sink(tf.filepath);
            store::value_copy(database, sink);
            for (std::string const& key : database.list()) {
                CHECK(database.get(key) == sink.get(key));
                CHECK(sink.get(key) != sink.null_value());
            }
            REQUIRE(tf.filepath.is_file());
            REQUIRE(tf.filepath.is_readable());
        }
        
        SECTION("[rocksdb] » Copy the Rocks database using value_copy() and xattr with a TemporaryName")
        {
            TemporaryName tn(".json");
            FileSink sink(tn.pathname);
            store::value_copy(database, sink);
            for (std::string const& key : database.list()) {
                CHECK(database.get(key) == sink.get(key));
                CHECK(sink.get(key) != sink.null_value());
            }
            REQUIRE(tn.pathname.is_file());
            REQUIRE(tn.pathname.is_readable());
        }
        
    }

} /// namespace (anon.)
