
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/attributes.h>
// #include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/filesystem/nowait.h>
#include <libimread/file.hh>
// #include <libimread/filehandle.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    // using filesystem::switchdir;
    // using filesystem::resolver;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    
    using filesystem::detail::nowait_t;
    using filesystem::detail::stringvec_t;
    // using filesystem::attribute::accessor_t;
    using filesystem::attribute::detail::nullstring;
    
    using im::FileSource;
    using im::FileSink;
    
    TEST_CASE("[attributes] xattr read with `path::xattr()` via `path::walk()`",
              "[xattr-read-path-walk-on-basedir]")
    {
        nowait_t nowait;
        
        path basedir = path(im::test::basedir).parent();
        // path basedir = path::home() / "Downloads";
        // path basedir = path::home();
        
        REQUIRE(basedir.exists());
        REQUIRE(basedir.is_directory());
        REQUIRE(basedir.is_readable());
        // switchdir s(basedir);
        
        basedir.walk([](path const& p,
                        stringvec_t& directories,
                        stringvec_t& files) {
            
            std::for_each(directories.begin(),
                          directories.end(),
                     [&p](std::string const& directory) {
                path ourdir = p/directory;
                
                REQUIRE(ourdir.is_directory());
                stringvec_t exes(ourdir.xattrs());
                
                if (exes.empty()) {
                    // std::cout << "> No xattrs found" << std::endl << std::endl;
                } else {
                    std::cout << "Directory: " << ourdir << std::endl;
                    std::cout << "> Found " << exes.size()
                              << " xattrs:" << std::endl;
                    std::for_each(exes.begin(),
                                  exes.end(),
                        [&ourdir](std::string const& x) {
                        // std::cout << "> " << x << " : "
                        //                   << ourdir.xattr(x) << std::endl;
                    });
                    // std::cout << std::endl;
                }
            });
            
            std::for_each(files.begin(),
                          files.end(),
                     [&p](std::string const& file) {
                path ourf = p/file;
                
                REQUIRE(ourf.is_file());
                stringvec_t exes(ourf.xattrs());
                
                if (exes.empty()) {
                    // std::cout << "> No xattrs found" << std::endl << std::endl;
                } else {
                    // std::cout << "File: " << ourf << std::endl;
                    // std::cout << "> Found " << exes.size()
                    //           << " xattrs:" << std::endl;
                    std::for_each(exes.begin(),
                                  exes.end(),
                          [&ourf](std::string const& x) {
                        // std::cout << "> " << x << " : "
                        //                   << ourf.xattr(x) << std::endl;
                    });
                    // std::cout << std::endl;
                }
            });
        
        });
    
    }
    
    TEST_CASE("[attributes] xattr read/write with `path::xattr()` via `TemporaryDirectory` and `NamedTemporaryFile`",
              "[xattr-read-write-temporarydirectory-namedtemporaryfile]")
    {
        TemporaryDirectory td("test-td");
        td.dirpath.xattr("yo-dogg", "I heard you like xattr writes");
        td.dirpath.xattr("dogg-yo", "So we put some strings in your strings");
        
        CHECK(td.dirpath.xattr("yo-dogg") == "I heard you like xattr writes");
        CHECK(td.dirpath.xattr("dogg-yo") == "So we put some strings in your strings");
        CHECK(td.dirpath.xattr("dogg-NO") == nullstring);
        
        CHECK(td.dirpath.xattrcount() == 2);
        stringvec_t exes(td.dirpath.xattrs());
        CHECK(!exes.empty());
        // std::cout << "> Found " << exes.size()
        //           << " xattrs:" << std::endl;
        std::for_each(exes.begin(),
                      exes.end(),
                  [&](std::string const& x) {
            // std::cout << "> " << x << " : "
            //                   << td.dirpath.xattr(x) << std::endl;
        });
        // std::cout << std::endl;
    }
    
    TEST_CASE("[attributes] xattr descriptor read/write via store API in `im::file_source_sink`",
              "[xattr-descriptor-read-write-store-api-file_source_sink]")
    {
        NamedTemporaryFile tf(".txt");
        {
            FileSource fdb(tf.filepath);
            fdb.set("yo-dogg", "I heard you like xattr writes");
            fdb.set("dogg-yo", "So we put some strings in your strings");
            
            CHECK(fdb.get("yo-dogg") == "I heard you like xattr writes");
            CHECK(fdb.get("dogg-yo") == "So we put some strings in your strings");
            CHECK(fdb.get("dogg-NO") == nullstring);
            
            CHECK(fdb.path().xattr("yo-dogg") == "I heard you like xattr writes");
            CHECK(fdb.path().xattr("dogg-yo") == "So we put some strings in your strings");
            CHECK(fdb.path().xattr("dogg-NO") == nullstring);
        }
        CHECK(tf.filepath.xattr("yo-dogg") == "I heard you like xattr writes");
        CHECK(tf.filepath.xattr("dogg-yo") == "So we put some strings in your strings");
        CHECK(tf.filepath.xattr("dogg-NO") == nullstring);
    }
    
} /// namespace (anon.)
