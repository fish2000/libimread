
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/JSON/json11.h>
// #include <libimread/ext/pystring.hh>

#include <libimread/file.hh>
// #include <libimread/filehandle.hh>
#include <libimread/store.hh>
#include <libimread/rocks.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::TemporaryName;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    using filesystem::detail::stringvec_t;
    using im::FileSource;
    using im::FileSink;
    
    TEST_CASE("[rocksdb] Create, update, read from, and ultimately destroy a RocksDB database",
              "[rocksdb-create-update-readfrom-ultimately-destroy-database]")
    {
        TemporaryDirectory td("test-rocks");                        /// the directory for all of this test’s genereated data
        path rockspth = td.dirpath.join("data.db");                 /// path to the on-disk storage for the Rocks instance
        path rocksjsonpth = td.dirpath.join("rocks-dump.json");     /// path to a JSON dump of the Rocks instance data
        path memoryjsonpth = td.dirpath.join("memory-dump.json");   /// path to a JSON dump of the in-memory stringmap data
        store::rocks database(rockspth.str());                      /// the actual Rocks “store” kv-interface database instance
        
        path basedir = path(im::test::basedir);                     /// the directory for all of this test’s read-only data
        path jsondir = basedir / "json";                            /// the read-only JSON data subdirectory
        path jsondoc = jsondir / "assorted.json";                   /// path to some random-ish JSON read-only test data
        
        std::size_t manualcount = 0;                                /// keep a manual master count of all database entries
        
        REQUIRE(jsondoc.is_readable());
        Json dictlist = Json::load(jsondoc.str());                  /// `assorted.json` contains a list of dicts,
        int max = dictlist.size();                                  /// each of which has a guaranteed-unique _id field
        
        for (int idx = 0; idx < max; ++idx) {                       /// manual loop over each dict in the list
            Json dict = dictlist.at(idx);
            std::string prefix(dict.get("_id"));
            
            for (std::string const& key : dict.keys()) {            /// iterate over the individual dict keys
                std::string nk = prefix + ":" + key;                /// construct a prefixed key for use in Rocks
                
                // WTF("KEY: ",    nk,
                //     "VALUE: ",  std::string(dict.get(key)));
                std::string nv = dict.cast<std::string>(key);
                CHECK(database.set(nk, nv));
                
                ++manualcount;                                      /// increment manual count
            }
        }
        
        REQUIRE(database.filepath() == rockspth.str());             /// ensure database paths are equal,
        REQUIRE(path::inode(database.filepath()) ==                 /// both as path literals and on-disk structures)
                rockspth.inode());
        
        REQUIRE(database.memorysize() > 0);                         /// ensure database takes up space in memory
        
        REQUIRE(database.count() == manualcount);                   /// ensure manual count and RocksDB internal count are equal
        
        for (std::string const& key : database.list()) {            /// ensure the database contains no null-value strings
            CHECK(database.get(key) != database.null_value());
        }
        
        store::stringmap memcopy(database);                         /// duplicate the RocksDB database to an in-memory stringmap
        
        REQUIRE(database.count() == memcopy.count());               /// ensure 1) the two databases hold the same number of values,
        for (std::string const& key : database.list()) {            /// 2) that the values themselves are equal,
            CHECK(database.get(key) == memcopy.get(key));           ///    and,
            CHECK(memcopy.get(key) != memcopy.null_value());        /// 3) that none of the values are std::string{ NULL_STR }.
        }
        
        // Json(database.mapping()).dump(rocksjsonpth.str());
        // Json(memcopy.mapping()).dump(memoryjsonpth.str());
        database.dump(rocksjsonpth);                                /// dump the databases to disk-based representations (currently JSON dicts)
        memcopy.dump(memoryjsonpth);
        
        REQUIRE(rocksjsonpth.exists());                             /// ensure the above calls to store::rocks::dump() do what they should
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
                if (key != reconstituted.null_key()) {
                    // WTF("KEY: ", key);
                    CHECK(database.get(key) == reconstituted.get(key));
                    CHECK(reconstituted.get(key) != reconstituted.null_value());
                }
            }
            
            CHECK(database == reconstituted);
            
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
