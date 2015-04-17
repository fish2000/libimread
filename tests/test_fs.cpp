
#include <unistd.h>
#include <string>
#include <vector>
#include <algorithm>
#include <libimread/libimread.hpp>
#include <libimread/fs.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

using im::fs::path;
using im::fs::switchdir;

TEST_CASE("Check if `basedir` is a directory", "[fs-basedir-isdirectory]") {
    path basedir(im::basedir);
    REQUIRE(basedir.is_directory());
}

TEST_CASE("Ensure `switchdir()` changes back on scope exit", "[fs-switchdir-change-back-scope-exit]") {
    path basedir(im::basedir);
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
    path basedir(im::basedir);
    std::vector<path> v = basedir.list("*.jpg");
    REQUIRE(v.size() == im::num_jpg);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::num_jpg; idx++) {
        REQUIRE((basedir/im::jpg[idx]).is_file());
    }
}

TEST_CASE("Check count of test jpeg files", "[fs-count-jpeg]") {
    path basedir(im::basedir);
    std::vector<path> v = basedir.list("*.jpeg");
    REQUIRE(v.size() == im::num_jpeg);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::num_jpeg; idx++) {
        REQUIRE((basedir/im::jpeg[idx]).is_file());
    }
}

TEST_CASE("Check count of test png files", "[fs-count-png]") {
    path basedir(im::basedir);
    std::vector<path> v = basedir.list("*.png");
    REQUIRE(v.size() == im::num_png);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::num_png; idx++) {
        REQUIRE((basedir/im::png[idx]).is_file());
    }
}

TEST_CASE("Check count of test tif files", "[fs-count-tif]") {
    path basedir(im::basedir);
    std::vector<path> v = basedir.list("*.tif");
    REQUIRE(v.size() == im::num_tif);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::num_tif; idx++) {
        REQUIRE((basedir/im::tif[idx]).is_file());
    }
}

TEST_CASE("Check count of test tiff files (should be 0)", "[fs-count-tiff]") {
    path basedir(im::basedir);
    std::vector<path> v = basedir.list("*.tiff");
    REQUIRE(v.size() == im::num_tiff);
    std::for_each(v.begin(), v.end(), [&](path &p){
        REQUIRE((basedir/p).is_file());
    });
    for (int idx = 0; idx < im::num_tiff; idx++) {
        REQUIRE((basedir/im::tiff[idx]).is_file());
    }
}

