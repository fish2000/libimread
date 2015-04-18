
#include <unistd.h>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include <libimread/libimread.hpp>
#include <libimread/fs.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

using im::fs::path;
using im::fs::switchdir;

TEST_CASE("Check if `basedir` is a directory", "[fs-basedir-isdirectory]") {
    path basedir(im::test::basedir);
    REQUIRE(basedir.is_directory());
}

TEST_CASE("Ensure `switchdir()` changes back on scope exit", "[fs-switchdir-change-back-scope-exit]") {
    path basedir(im::test::basedir);
    path absdir(basedir.make_absolute());
    path tmpdir("/private/tmp");
    REQUIRE(absdir.is_directory());
    REQUIRE(tmpdir.is_directory());
    const char *current = path::cwd();
    chdir(absdir);
    REQUIRE(path::cwd().make_absolute().str() == absdir.str());
    {
        switchdir s(tmpdir);
        REQUIRE(path::cwd().make_absolute().str() == tmpdir.str());
    }
    REQUIRE(path::cwd().make_absolute().str() == absdir.str());
    chdir(current);
}

TEST_CASE("Check count and names of test jpg files", "[fs-check-jpg]") {
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

TEST_CASE("Check count of test jpeg files", "[fs-count-jpeg]") {
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

TEST_CASE("Check count of test png files", "[fs-count-png]") {
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

TEST_CASE("Check count of test tif files", "[fs-count-tif]") {
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

TEST_CASE("Check count of test tiff files (should be 0)", "[fs-count-tiff]") {
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

TEST_CASE("Count both jpg and jpeg files with a regex", "[fs-count-jpg-jpeg-regex]") {
    path basedir(im::test::basedir);
    std::regex re("(jpg|jpeg)$", RE_FLAGS);
    std::vector<path> v = basedir.list(re);
    REQUIRE(v.size() == im::test::num_jpg + im::test::num_jpeg);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
}

