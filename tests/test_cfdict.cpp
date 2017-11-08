
#define CATCH_CONFIG_FAST_COMPILE

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/JSON/json11.h>
// #include <libimread/ext/pystring.hh>

#include <libimread/file.hh>
#include <libimread/store.hh>
#include <libimread/corefoundation.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::TemporaryName;
    using filesystem::TemporaryDirectory;
    using filesystem::detail::stringvec_t;
    using im::FileSource;
    using im::FileSink;
    using im::detail::cfp_t;
    
    TEST_CASE("[cfdict] Create, update, read from, and ultimately destroy a CFDictionaryRef store",
              "[cfdict-create-update-read-from-ultimately-destroy-cfdictionaryref-store]")
    {
        TemporaryDirectory td("test-cfdict");                       /// the directory for all of this test’s genereated data
        path cfdictjsonpth = td.dirpath.join("cfdict-dump.json");   /// path to a JSON dump of the CFDictionaryRef data
        path memoryjsonpth = td.dirpath.join("memory-dump.json");   /// path to a JSON dump of the in-memory stringmap data
        store::cfdict database;                                     /// the actual CFDictionaryRef “store” kv-interface wrapper instance
        
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
                std::string nk = prefix + ":" + key;                /// construct a prefixed key for use in the dictionary
                
                // WTF("KEY: ",    nk,
                //     "VALUE: ",  std::string(dict.get(key)));
                std::string nv = dict.cast<std::string>(key);
                CHECK(database.set(nk, nv));
                
                ++manualcount;                                      /// increment manual count
            }
        }
        
        REQUIRE(database.count() == manualcount);                   /// ensure manual count and CFDictionaryRef internal count are equal
        
        for (std::string const& key : database.list()) {            /// ensure the database contains no null-value strings
            CHECK(database.get(key) != database.null_value());
        }
        
        store::stringmap memcopy(database);                         /// duplicate the dict-backed database to an in-memory stringmap
        
        REQUIRE(database.count() == memcopy.count());               /// ensure 1) the two databases hold the same number of values,
        for (std::string const& key : database.list()) {            /// 2) that the values themselves are equal,
            CHECK(database.get(key) == memcopy.get(key));           ///    and,
            CHECK(memcopy.get(key) != memcopy.null_value());        /// 3) that none of the values are std::string{ NULL_STR }.
        }
        
        store::stringmap memcopy2(database);                        /// duplicate the dict-backed database to an in-memory stringmap
        
        REQUIRE(database.count() == memcopy2.count());              /// ensure 1) the two databases hold the same number of values,
        for (std::string const& key : database.list()) {            /// 2) that the values themselves are equal,
            CHECK(database.get(key) == memcopy2.get(key));          ///    and,
            CHECK(memcopy.get(key) != memcopy2.null_value());       /// 3) that none of the values are std::string{ NULL_STR }.
        }
        
        database.dump(cfdictjsonpth);                               /// dump the databases to disk-based representations (currently JSON dicts)
        memcopy.dump(memoryjsonpth);
        
        REQUIRE(cfdictjsonpth.exists());                            /// ensure the above calls to store::cfdict::dump() do what they should
        REQUIRE(cfdictjsonpth.is_file());
        REQUIRE(memoryjsonpth.exists());
        REQUIRE(memoryjsonpth.is_file());
        
        cfp_t<CFDictionaryRef> immutable(const_cast<__CFDictionary *>(
                                         database.dictionary()));   /// copy the databases’ internal CFMutableDictionaruyRef into
                                                                    /// a managed-pointer non-mutable CFDictionaryRef instance
        
        REQUIRE(database.count() ==
                CFDictionaryGetCount(immutable.get()));
        
        SECTION("[cfdict] » Reconstitute the CFDictionaryRef database, "
                           "using store::stringmap::load()")
        {
            // WTF("PRE-CONSTITUTED FILE SIZE: ", cfdictjsonpth.filesize());
            
            store::stringmap reconstituted = store::stringmap::load(cfdictjsonpth);
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
        }
        
        SECTION("[cfdict] » Reconstitute the CFDictionaryRef database, "
                           "using store::cfdict::load()")
        {
            // WTF("PRE-CONSTITUTED FILE SIZE: ", cfdictjsonpth.filesize());
            
            store::cfdict reconstituted = store::cfdict::load(cfdictjsonpth);
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
        }
        
        SECTION("[cfdict] » Copy the dict-backed database, "
                           "using store::value_copy(), "
                           "and xattr() with a TemporaryName RAII struct")
        {
            /// Allocate character-writeable stream
            TemporaryName tn(".json");
            FileSink sink(tn.pathname);
            
            /// Copy attributes to file sink
            store::value_copy(database, sink);
            
            /// Verify copied attributes
            for (std::string const& key : database.list()) {
                CHECK(database.get(key) == sink.get(key));
                CHECK(sink.get(key) != sink.null_value());
            }
            
            REQUIRE(tn.pathname.is_file());
            REQUIRE(tn.pathname.is_readable());
        }
        
        SECTION("[cfdict] » Transfer the dict-backed database, "
                           "using store::{prefix,defix}_copy(), "
                           "and xattr() alongside TemporaryName RAII structs")
        {
            /// Allocate two writeable FileSinks
            TemporaryName tn0(".json");
            TemporaryName tn1(".json");
            FileSink sink0(tn0.pathname);
            FileSink sink1(tn1.pathname);
            
            /// Copy attributes to first sink, adding a prefix
            store::prefix_copy(database, sink0, "prefix");
            
            /// Copy attributes to second sink, removing prefix
            store::defix_copy(sink0, sink1, "prefix");
            
            /// Verify copied attributes
            for (std::string const& key : database.list()) {
                CHECK(database.get(key) != sink0.get(key));
                CHECK(database.get(key) == sink1.get(key));
                CHECK(sink1.get(key) != sink1.null_value());
            }
            
            REQUIRE(tn0.pathname.is_file());
            REQUIRE(tn0.pathname.is_readable());
            REQUIRE(tn1.pathname.is_file());
            REQUIRE(tn1.pathname.is_readable());
        }
        
    }
    
}