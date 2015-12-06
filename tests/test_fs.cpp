
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

#include <libimread/libimread.hpp>
#include <libimread/fs.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

using im::fs::path;
using im::fs::switchdir;
using im::fs::resolver;

TEST_CASE("[filesystem] Check if `basedir` is a directory",
          "[fs-basedir-isdirectory]") {
    path basedir(im::test::basedir);
    REQUIRE(basedir.is_directory());
}

TEST_CASE("[filesystem] Ensure `switchdir()` changes back on scope exit",
          "[fs-switchdir-change-back-scope-exit]") {
    path basedir(im::test::basedir);
    path absdir(basedir.make_absolute());
    path tmpdir("/private/tmp");
    /// preflight: ensure that `absdir` (which is `basedir` in a suit)
    ///            and `tmpdir` (which guess) are actually existant and valid
    REQUIRE(absdir.is_directory());
    REQUIRE(tmpdir.is_directory());
    /// preflight: store the current WD using path::cwd(),
    ///            manually call chdir() to start out in `basedir`
    const char *current = path::cwd();
    chdir(absdir);
    
    /// confirm the new WD using path::cwd() and operator==() with `absdir`
    bool check_one = bool(path::cwd() == absdir);
    REQUIRE(check_one);
    
    {
        /// switch working directory to /private/tmp
        switchdir s(tmpdir);
        /// confirm we are in the new directory
        bool check_two = bool(path::cwd() == tmpdir);
        /// confirm we came from the old directory
        bool check_two_and_a_half = bool(s.from() == absdir);
        /// confirm our new location's listing contains at least one file
        bool check_two_and_two_thirds = bool(path::cwd().list().size() > 0);
        /// proceed ...
        REQUIRE(check_two);
        REQUIRE(check_two_and_a_half);
        REQUIRE(check_two_and_two_thirds);
        /// ... aaaand the working directory flips back to `basedir` at scope exit
    }
    
    /// confirm we are back in `basedir`
    bool check_three = bool(path::cwd() == absdir);
    REQUIRE(check_three);
    
    /// post-hoc: manually chdir() back to point of origin,
    ///           so as to hopefully return back into the test runner without having
    ///           problematically fucked with its state (like at least as minimally
    ///           fucked as one can, like possibly, erm)
    chdir(current);
}

TEST_CASE("[filesystem] Test path::hash() and std::hash<path> specialization integrity",
          "[fs-path-hash-and-std-hash-specialization-integrity]") {
    path basedir(im::test::basedir);
    path absdir(basedir.make_absolute());
    path tmpdir("/private/tmp");
    std::hash<path> hasher;
    /// the test data header-generator will write 'basedir' out as absolute
    REQUIRE(basedir == absdir);
    REQUIRE(basedir != tmpdir);
    /// path::operator==() uses path::hash()
    REQUIRE(basedir.hash() == absdir.hash());
    REQUIRE(basedir.hash() != tmpdir.hash());
    /// std::hash<path> also uses path::hash()
    REQUIRE(hasher(basedir) == basedir.hash());
    REQUIRE(hasher(basedir) != tmpdir.hash());
    REQUIRE(hasher(basedir) == hasher(absdir));
    /// path::hash<P>(p) forwards to path::hash()
    REQUIRE(path::hash(basedir) == path::hash(absdir));
    REQUIRE(path::hash(basedir) != path::hash(tmpdir));
    REQUIRE(path::hash(basedir) == absdir.hash());
    REQUIRE(path::hash(basedir) != tmpdir.hash());
    REQUIRE(path::hash(basedir) == hasher(absdir));
    REQUIRE(path::hash(basedir) != hasher(tmpdir));
}

TEST_CASE("[filesystem] Check count and names of test jpg files",
          "[fs-check-jpg]") {
    path basedir(im::test::basedir);
    std::vector<path> v = basedir.list("*.jpg");
    REQUIRE(v.size() == im::test::num_jpg);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::test::num_jpg; idx++) {
        REQUIRE((basedir/im::test::jpg[idx]).is_file());
    }
}

TEST_CASE("[filesystem] Test `path::walk()` on `im::test::basedir`",
          "[fs-path-walk-on-basedir]") {
    path basedir(im::test::basedir);
    basedir.walk([](const path& p,
                    std::vector<std::string>& directories,
                    std::vector<std::string>& files) {
        std::for_each(directories.begin(), directories.end(), [&](const std::string& d) {
            std::cout << "Directory: " << p/d << std::endl;
            REQUIRE((p/d).is_directory());
        });
        std::for_each(files.begin(), files.end(), [&](const std::string& f) {
            std::cout << "File: " << p/f << std::endl;
            REQUIRE((p/f).is_file());
        });
    });
    
}

TEST_CASE("[filesystem] Check count of test jpeg files",
          "[fs-count-jpeg]") {
    path basedir(im::test::basedir);
    std::vector<path> v = basedir.list("*.jpeg");
    REQUIRE(v.size() == im::test::num_jpeg);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::test::num_jpeg; idx++) {
        REQUIRE((basedir/im::test::jpeg[idx]).is_file());
    }
}

TEST_CASE("[filesystem] Check count of test png files",
          "[fs-count-png]") {
    path basedir(im::test::basedir);
    std::vector<path> v = basedir.list("*.png");
    REQUIRE(v.size() == im::test::num_png);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::test::num_png; idx++) {
        REQUIRE((basedir/im::test::png[idx]).is_file());
    }
}

TEST_CASE("[filesystem] Check count of test tif files",
          "[fs-count-tif]") {
    path basedir(im::test::basedir);
    std::vector<path> v = basedir.list("*.tif");
    REQUIRE(v.size() == im::test::num_tif);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::test::num_tif; idx++) {
        REQUIRE((basedir/im::test::tif[idx]).is_file());
    }
}

TEST_CASE("[filesystem] Check count of test tiff files (should be 0)",
          "[fs-count-tiff]") {
    path basedir(im::test::basedir);
    std::vector<path> v = basedir.list("*.tiff");
    REQUIRE(v.size() == im::test::num_tiff);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::test::num_tiff; idx++) {
        REQUIRE((basedir/im::test::tiff[idx]).is_file());
    }
}

#define RE_FLAGS std::regex::extended | std::regex::icase

TEST_CASE("[filesystem] Count both jpg and jpeg files with a regex",
          "[fs-count-jpg-jpeg-regex]") {
    path basedir(im::test::basedir);
    std::regex re("(jpg|jpeg)$", RE_FLAGS);
    std::vector<path> v = basedir.list(re);
    REQUIRE(v.size() == im::test::num_jpg + im::test::num_jpeg);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
}

TEST_CASE("[filesystem] Test the system path resolver",
          "[fs-system-path-resolver]") {
    resolver syspaths = resolver::system();
    REQUIRE(syspaths.resolve(path("ls")) != path());
    REQUIRE(syspaths.resolve(path("clang")) != path());
    REQUIRE(syspaths.resolve(path("YoDogg")) == path());
}


