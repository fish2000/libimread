/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <glob.h>
#include <algorithm>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    directory ddopen(const char *c) {
        return directory(opendir(path::absolute(c)));
    }
    directory ddopen(const std::string &s) {
        return directory(opendir(path::absolute(s)));
    }
    directory ddopen(const path &p) {
        return directory(opendir(p.make_absolute().c_str()));
    }
    
    inline const char *fm(mode m) { return m == mode::READ ? "r+b" : "w+x"; }
    
    // file ffopen(const char *c, mode m) {
    //     return file(fopen(c, fm(m)));
    // }
    file ffopen(const std::string &s, mode m) {
        return file(fopen(s.c_str(), fm(m)));
    }
    // file ffopen(const path &p, mode m) {
    //     return file(fopen(p.c_str(), fm(m)));
    // }
    
    bool path::match(const std::regex &pattern, bool case_sensitive) {
        return std::regex_match(str(), pattern);
    }
    
    bool path::search(const std::regex &pattern, bool case_sensitive) {
        return std::regex_search(str(), pattern);
    }
    
    std::vector<path> path::list(bool full_paths) {
        /// list all files
        if (!is_directory()) {
            imread_raise(FileSystemError,
                "Can't list files from a non-directory:", str());
        }
        path abspath = make_absolute();
        std::vector<path> out;
        {
            directory d = ddopen(abspath);
            if (!d.get()) {
                imread_raise(FileSystemError,
                    "Internal error in opendir():", strerror(errno));
            }
            struct dirent *entp;
            while ((entp = readdir(d.get())) != NULL) {
                if (entp->d_type == DT_DIR || entp->d_type == DT_REG || entp->d_type == DT_LNK) {
                    /// ... it's either a directory, a regular file, or a symbolic link
                    out.push_back(full_paths ? abspath/entp->d_name : path(entp->d_name));
                }
            }
        } /// scope exit for d
        return out;
    }
    
    static const int glob_pattern_flags = GLOB_TILDE | GLOB_NOSORT | GLOB_DOOFFS;
    
    std::vector<path> path::list(const char *pattern, bool full_paths) {
        /// list files with glob
        if (!pattern) {
            imread_raise(FileSystemError,
                "No pattern provided for listing:", str());
        }
        if (!is_directory()) {
            imread_raise(FileSystemError,
                "Can't list files from a non-directory:", str());
        }
        path abspath = make_absolute();
        glob_t g = {0};
        {
            filesystem::switchdir s(abspath);
            glob(pattern, glob_pattern_flags, NULL, &g);
        }
        std::vector<path> out;
        for (std::size_t idx = 0; idx != g.gl_pathc; ++idx) {
            out.push_back(full_paths ? abspath/g.gl_pathv[idx] : path(g.gl_pathv[idx]));
        }
        globfree(&g);
        return out;
    }
    
    std::vector<path> path::list(const std::string &pattern, bool full_paths) {
        return list(pattern.c_str());
    }
    
    static const int regex_flags        = std::regex::extended;
    static const int regex_flags_icase  = std::regex::extended | std::regex::icase;
    
    std::vector<path> path::list(const std::regex &pattern, bool case_sensitive, bool full_paths) {
        /// list files with regex object
        if (!is_directory()) {
            imread_raise(FileSystemError,
                "Can't list files from a non-directory:", str());
        }
        path abspath = make_absolute();
        std::vector<path> unfiltered = abspath.list();
        std::vector<path> out;
        std::for_each(unfiltered.begin(), unfiltered.end(), [&](path &p) {
            if (p.search(pattern, case_sensitive)) {
                out.push_back(full_paths ? abspath/p : p);
            }
        });
        return out;
    }

}