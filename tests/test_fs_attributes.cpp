
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/attributes.h>
// #include <libimread/ext/filesystem/directory.h>
// #include <libimread/ext/filesystem/resolver.h>
#include <libimread/ext/filesystem/temporary.h>
// #include <libimread/file.hh>
// #include <libimread/filehandle.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    // using filesystem::switchdir;
    // using filesystem::resolver;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    
    using filesystem::detail::stringvec_t;
    using filesystem::attribute::accessor_t;
    using filesystem::attribute::detail::nullstring;
    
    TEST_CASE("[attributes] xattr read with `path::xattr()` via `path::walk()`",
              "[xattr-read-path-walk-on-basedir]") {
        path basedir(im::test::basedir);
        basedir.walk([](path const& p,
                        stringvec_t& directories,
                        stringvec_t& files) {
            std::for_each(directories.begin(), directories.end(), [&](std::string const& d) {
                std::cout << "Directory: " << p/d << std::endl;
                REQUIRE((p/d).is_directory());
                stringvec_t exes((p/d).xattrs());
                if (exes.empty()) {
                    std::cout << "> No xattrs found" << std::endl;
                } else {
                    std::cout << "> Found " << exes.size() << " xattrs:" << std::endl;
                    std::for_each(exes.begin(), exes.end(), [&](std::string const& x) {
                        std::cout << "> " << x << " : " << (p/d).xattr(x) << std::endl;
                    });
                }
            });
            std::for_each(files.begin(), files.end(), [&](std::string const& f) {
                std::cout << "File: " << p/f << std::endl;
                REQUIRE((p/f).is_file());
                stringvec_t exes((p/f).xattrs());
                if (exes.empty()) {
                    std::cout << "> No xattrs found" << std::endl;
                } else {
                    std::cout << "> Found " << exes.size() << " xattrs:" << std::endl;
                    std::for_each(exes.begin(), exes.end(), [&](std::string const& x) {
                        std::cout << "> " << x << " : " << (p/f).xattr(x) << std::endl;
                    });
                }
            });
        });
    
    }
    
    TEST_CASE("[attributes] xattr read/write with `path::xattr()` via `TemporaryDirectory` and `NamedTemporaryFile`",
              "[xattr-read-write-temporarydirectory-namedtemporaryfile]") {
        TemporaryDirectory td("test-td");
        td.dirpath.xattr("yo-dogg", "I heard you like xattr writes");
        td.dirpath.xattr("dogg-yo", "So we put some strings in your strings");
        CHECK(td.dirpath.xattr("yo-dogg") == "I heard you like xattr writes");
        CHECK(td.dirpath.xattr("dogg-yo") == "So we put some strings in your strings");
        CHECK(td.dirpath.xattr("dogg-NO") == nullstring);
    }
    
} /// namespace (anon.)
