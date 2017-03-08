
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/resolver.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/ext/filesystem/execute.h>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    // using filesystem::dotpath;
    using filesystem::switchdir;
    using filesystem::resolver;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    
    TEST_CASE("[filesystem] Check if `basedir` is a directory",
              "[fs-basedir-isdirectory]") {
        path basedir(im::test::basedir);
        REQUIRE(basedir.is_directory());
    }
    
    TEST_CASE("[filesystem] Ensure `switchdir()` changes back on scope exit",
              "[fs-switchdir-change-back-scope-exit]") {
        path basedir(im::test::basedir);
        path absdir(basedir.make_absolute());
        path tmpdir(path::tmp());
        
        /// preflight: ensure that `absdir` (which is `basedir` in a suit)
        ///            and `tmpdir` (which guess) are actually existant and valid
        REQUIRE(absdir.is_directory());
        REQUIRE(tmpdir.is_directory());
        
        /// preflight: store the current WD using path::cwd(),
        ///            manually call chdir() to start out in `basedir`
        const char* current = path::cwd().c_str();
        ::chdir(absdir);
        
        /// confirm the new WD using path::cwd() and operator==() with `absdir`
        bool check_one = bool(path::cwd() == absdir);
        REQUIRE(check_one);
        
        {
            /// switch working directory to tmpdir
            switchdir s(tmpdir);
            /// confirm we are in the new directory
            bool check_two = bool(path::cwd() == tmpdir);
            // WTF("CWD: ", path::cwd().str());
            // WTF("TMP: ", tmpdir.str());
            /// confirm we came from the old directory
            bool check_two_and_a_half = bool(s.from() == absdir);
            /// confirm our new location's listing contains at least one file
            bool check_two_and_two_thirds = bool(path::cwd().list().size() > 0);
            /// proceed ...
            // REQUIRE(check_two);
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
        ::chdir(current);
        // REQUIRE(path::cwd().c_str() == current);
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
        std::for_each(v.begin(), v.end(), [&](path& p) {
            REQUIRE((basedir/p).is_file());
        });
        for (int idx = 0; idx < im::test::num_jpg; idx++) {
            REQUIRE((basedir/im::test::jpg[idx]).is_file());
        }
    }
    
    TEST_CASE("[filesystem] Test `path::walk()` on `im::test::basedir`",
              "[fs-path-walk-on-basedir]") {
        path basedir(im::test::basedir);
        basedir.walk([](path const& p,
                        std::vector<std::string>& directories,
                        std::vector<std::string>& files) {
            std::for_each(directories.begin(), directories.end(), [&](std::string const& d) {
                std::cout << "Directory: " << p/d << std::endl;
                REQUIRE((p/d).is_directory());
            });
            std::for_each(files.begin(), files.end(), [&](std::string const& f) {
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
        std::for_each(v.begin(), v.end(), [&](path& p) {
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
        std::for_each(v.begin(), v.end(), [&](path& p) {
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
        std::for_each(v.begin(), v.end(), [&](path& p) {
            REQUIRE((basedir/p).is_file());
        });
        for (int idx = 0; idx < im::test::num_tif; idx++) {
            REQUIRE((basedir/im::test::tif[idx]).is_file());
        }
    }
    
    TEST_CASE("[filesystem] Check count of test tiff files",
              "[fs-count-tiff]") {
        path basedir(im::test::basedir);
        std::vector<path> v = basedir.list("*.tiff");
        REQUIRE(v.size() == im::test::num_tiff);
        std::for_each(v.begin(), v.end(), [&](path& p) {
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
        std::for_each(v.begin(), v.end(), [&](path& p) {
            REQUIRE((basedir/p).is_file());
        });
    }
    
    TEST_CASE("[filesystem] Check count of test pvr files",
              "[fs-count-pvr]") {
        path basedir(im::test::basedir);
        std::vector<path> v = basedir.list("*.pvr");
        REQUIRE(v.size() == im::test::num_pvr);
        std::for_each(v.begin(), v.end(), [&](path& p) {
            REQUIRE((basedir/p).is_file());
        });
        for (int idx = 0; idx < im::test::num_pvr; idx++) {
            REQUIRE((basedir/im::test::pvr[idx]).is_file());
        }
    }
    
    TEST_CASE("[filesystem] Test the system path resolver",
              "[fs-system-path-resolver]") {
        resolver syspaths = resolver::system();
        
        CHECK(syspaths.resolve("ls") != path());
        CHECK(syspaths.resolve("clang") != path());
        CHECK(syspaths.resolve("YoDogg") == path());
        
        CHECK(syspaths.contains("ls"));
        CHECK(syspaths.contains("clang"));
        CHECK(!syspaths.contains("YoDogg"));
        
        WTF("SYSTEM PATHS: ",
            "", syspaths.to_string("\n\t"),
            "", FF("RESOLVER SIZE: %i", syspaths.size()));
    }
    
    TEST_CASE("[filesystem] Test the default resolver with path::executable()",
              "[fs-default-resolver-with-path-executable]") {
        path executable = path::executable();
        resolver dres(executable.parent());
        CHECK(dres.contains(executable.basename()));
        CHECK(dres.contains(path::currentprogram()));
    }
    
    TEST_CASE("[filesystem] Test the TemporaryDirectory RAII struct",
              "[fs-temporarydirectory-raii]")
    {
        TemporaryDirectory td("test-td");
        (td.dirpath/"test-td-subdir0").makedir();
        (td.dirpath/"test-td-subdir1").makedir();
        REQUIRE((td.dirpath/"test-td-subdir0").is_listable());
        REQUIRE((td.dirpath/"test-td-subdir1").is_listable());
    }
    
    TEST_CASE("[filesystem] Test the NamedTemporaryFile RAII struct's stream interface",
              "[fs-namedtemporaryfile-raii-stream-interface]")
    {
        NamedTemporaryFile tf(".txt");
        CHECK(tf.open());
        tf.stream << "Yo Dogg" << std::endl;
        tf.stream << "I Heard You Like" << std::endl;
        tf.stream << "The C++ Standard Library's" << std::endl;
        tf.stream << "Weird-Ass File I/O Interface" << std::endl;
        CHECK(tf.close());
    }
    
    TEST_CASE("[filesystem] Test path::extension(), path::extensions(), path::strip_extension(), and path::strip_extensions()",
              "[fs-test-path-extension-path-extensions-path-strip_extension-path-strip_extensions]") {
        path basedir(im::test::basedir);
        const std::vector<path> v = basedir.list(std::string("*.jpg"));
        REQUIRE(v.size() == im::test::num_jpg);
        std::for_each(v.begin(), v.end(), [&](path const& p) {
            CHECK((basedir/p).is_file());
            
            path oldpth(basedir/p);
            path newpth(oldpth + ".jpeg");
            
            CHECK(oldpth.extension() == "jpg");
            CHECK(newpth.extension() == "jpeg");
            CHECK(oldpth.extensions() == "jpg");
            CHECK(newpth.extensions() == "jpg.jpeg");
            
            path oldstrip    = oldpth.strip_extension();
            path oldstripped = oldpth.strip_extensions();
            path newstrip    = newpth.strip_extension();
            path newstripped = newpth.strip_extensions();
            
            CHECK(oldstrip    == oldstripped);
            CHECK(newstrip    == oldpth);
            CHECK(newstripped == oldpth.strip_extension());
            
            CHECK(newpth.split_extensions().size() == 2);
        });
    }
    
    TEST_CASE("[filesystem] Test the path::makedir_p() method",
              "[fs-test-path-makedir_p]")
    {
        TemporaryDirectory td("test-td");
        // path dp(td.dirpath.str());
        // path da(td.dirpath.str());
        path dp = td.dirpath;
        path da = td.dirpath;
        
        /// makedir_p() called on rvalue,
        /// with multiple target directories:
        REQUIRE((td.dirpath/"yo"/"dogg"/"i-heard"/"you-like"/"directories").makedir_p());
        CHECK((td.dirpath/"yo").is_directory());
        CHECK((td.dirpath/"yo"/"dogg").is_directory());
        CHECK((td.dirpath/"yo"/"dogg"/"i-heard").is_directory());
        CHECK((td.dirpath/"yo"/"dogg"/"i-heard"/"you-like").is_directory());
        CHECK((td.dirpath/"yo"/"dogg"/"i-heard"/"you-like"/"directories").is_directory());
        
        /// makedir_p() called on lvalue,
        /// with multiple target directories:
        dp /= "yo-yo";
        dp /= "dogg";
        dp /= "i-heard";
        dp /= "you-like";
        dp /= "directories";
        // WTF("ADJOINED DIRECTORY: ", dp.str());
        REQUIRE(dp.makedir_p());
        CHECK((td.dirpath/"yo-yo").is_directory());
        CHECK((td.dirpath/"yo-yo"/"dogg").is_directory());
        CHECK((td.dirpath/"yo-yo"/"dogg"/"i-heard").is_directory());
        CHECK((td.dirpath/"yo-yo"/"dogg"/"i-heard"/"you-like").is_directory());
        CHECK((td.dirpath/"yo-yo"/"dogg"/"i-heard"/"you-like"/"directories").is_directory());
        
        /// makedir_p() called on rvalue,
        /// with only one target directory:
        REQUIRE((td.dirpath/"yo_dogg_i_heard_you_like_directories").makedir_p());
        CHECK((td.dirpath/"yo_dogg_i_heard_you_like_directories").is_directory());
        
        /// makedir_p() called on lvalue,
        /// with only one target directory:
        da /= "yo";
        da += "_yo_dogg_i_heard_you_like_directories";
        // WTF("EXTENDED DIRECTORY: ", da.str());
        REQUIRE(da.makedir_p());
        CHECK((td.dirpath/"yo_yo_dogg_i_heard_you_like_directories").is_directory());
    }
    
    // TEST_CASE("[filesystem] Test dotpath subclass",
    //           "[fs-test-dotpath-subclass]") {
    //     dotpath basepath("ost.basepath.ClassName:YoDogg");
    //     dotpath parentpath("ost.basepath");
    //     CHECK(basepath.extension() == "YoDogg");
    //     CHECK(basepath.parent() == parentpath);
    // }
    
} /// namespace (anon.)
