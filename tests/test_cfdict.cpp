
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/JSON/json11.h>
// #include <libimread/ext/pystring.hh>

#include <libimread/file.hh>
// #include <libimread/filehandle.hh>
#include <libimread/store.hh>
#include <libimread/coregraphics.hh>

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
    
    TEST_CASE("[cfdict] Create, update, read from, and ultimately destroy a CFDictionaryRef store",
              "[cfdict-create-update-readfrom-ultimately-destroy-cfdictionaryref-store]")
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
                std::string nk = prefix + ":" + key;                /// construct a prefixed key for use in Rocks
                
                // WTF("KEY: ",    nk,
                //     "VALUE: ",  std::string(dict.get(key)));
                std::string nv = dict.cast<std::string>(key);
                CHECK(database.set(nk, nv));
                
                ++manualcount;                                      /// increment manual count
            }
        }
        
        REQUIRE(database.count() == manualcount);                   /// ensure manual count and CFDictionaryRef internal count are equal
    }
    
}