
#define CATCH_CONFIG_FAST_COMPILE

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

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    // using filesystem::switchdir;
    using filesystem::TemporaryName;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    
    using filesystem::detail::nowait_t;
    using filesystem::detail::stringvec_t;
    // using filesystem::detail::copyfile;
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
                
                if (!exes.empty()) {
                    std::for_each(exes.begin(),
                                  exes.end(),
                        [&ourdir](std::string const& x) {
                            CHECK(ourdir.xattr(x) != nullstring);
                    });
                }
            });
            
            std::for_each(files.begin(),
                          files.end(),
                     [&p](std::string const& file) {
                path ourf = p/file;
                
                REQUIRE(ourf.is_file());
                stringvec_t exes(ourf.xattrs());
                
                if (!exes.empty()) {
                    std::for_each(exes.begin(),
                                  exes.end(),
                          [&ourf](std::string const& x) {
                              CHECK(ourf.xattr(x) != nullstring);
                    });
                }
            });
        
        });
    
    }
    
    TEST_CASE("[attributes] xattr read/write with `path::xattr()` via `TemporaryDirectory`",
              "[xattr-read-write-path-temporarydirectory]")
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
        std::for_each(exes.begin(),
                      exes.end(),
                  [&](std::string const& x) {
                      CHECK(td.dirpath.xattr(x) != nullstring);
        });
    }
    
    TEST_CASE("[attributes] xattr descriptor read/write via store API in `im::file_source_sink` using `NamedTemporaryFile`",
              "[xattr-descriptor-read-write-store-api-file_source_sink-namedtemporaryfile]")
    {
        NamedTemporaryFile tf(".txt");
        
        tf.open();
        tf.stream << "YO DOGG";
        tf.close();
        
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
        
        {
            TemporaryName tn(".md");
            std::string source = tf.filepath.str();
            std::string destination = tn.pathname.str();
            
            REQUIRE(tf.filepath.exists());
            REQUIRE(tf.filepath.filesize() > 0);
            filesystem::detail::copyfile(source.c_str(),
                                         destination.c_str(),
                                         true); /// copy_attributes = true
            
            REQUIRE(tn.pathname.exists());
            CHECK(tn.pathname.xattr("yo-dogg") == "I heard you like xattr writes");
            CHECK(tn.pathname.xattr("dogg-yo") == "So we put some strings in your strings");
            CHECK(tn.pathname.xattr("dogg-NO") == nullstring);
            CHECK(path::xattrget(destination, "yo-dogg") == "I heard you like xattr writes");
            CHECK(path::xattrget(destination, "dogg-yo") == "So we put some strings in your strings");
            CHECK(path::xattrget(destination, "dogg-NO") == nullstring);
        }
        
        CHECK(tf.filepath.xattr("yo-dogg") == "I heard you like xattr writes");
        CHECK(tf.filepath.xattr("dogg-yo") == "So we put some strings in your strings");
        CHECK(tf.filepath.xattr("dogg-NO") == nullstring);
        
        CHECK(tf.filepath.xattrcount() == 2);
        stringvec_t exes(tf.filepath.xattrs());
        CHECK(!exes.empty());
        std::for_each(exes.begin(),
                      exes.end(),
                  [&](std::string const& x) {
                      CHECK(tf.filepath.xattr(x) != nullstring);
        });
    }
    
} /// namespace (anon.)
