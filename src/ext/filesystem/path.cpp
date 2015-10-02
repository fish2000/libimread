/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <sys/stat.h>
#include <glob.h>
#include <fcntl.h>
#include <cstdlib>
#include <algorithm>
#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    namespace detail {
        
        using stat_t = struct stat;
        using dirent_t = struct dirent;
        
        filesystem::directory ddopen(const char *c) {
            return filesystem::directory(::opendir(path::absolute(c).c_str()));
        }
        
        filesystem::directory ddopen(const std::string &s) {
            return filesystem::directory(::opendir(path::absolute(s).c_str()));
        }
        
        filesystem::directory ddopen(const path &p) {
            return filesystem::directory(::opendir(p.make_absolute().c_str()));
        }
        
        inline const char *fm(mode m) noexcept { return m == mode::READ ? "r+b" : "w+x"; }
        
        filesystem::file ffopen(const std::string &s, mode m) {
            return filesystem::file(fopen(s.c_str(), fm(m)));
        }
        
        const char *tmpdir() noexcept {
            /// cribbed/tweaked from boost
            const char *dirname;
            dirname = std::getenv("TMPDIR");
            if (NULL == dirname) { dirname = std::getenv("TMP"); }
            if (NULL == dirname) { dirname = std::getenv("TEMP"); }
            if (NULL == dirname) { dirname = "/tmp"; }
            return dirname;
        }
        
    }
    
    path::path(int descriptor) {
        char fdpath[PATH_MAX];
        int return_value = ::fcntl(descriptor, F_GETPATH, fdpath);
        if (return_value != -1) {
            set(fdpath);
        } else {
            imread_raise(FileSystemError,
                "Internal error: -1 returned by ::fnctl(descriptor, F_GETPATH, fdpath)",
                "where fdpath = ", fdpath);
        }
    }
    
    bool path::match(const std::regex &pattern, bool) const {
        return std::regex_match(str(), pattern);
    }
    
    bool path::search(const std::regex &pattern, bool) const {
        return std::regex_search(str(), pattern);
    }
    
    path path::make_absolute() const {
        char temp[PATH_MAX];
        if (::realpath(c_str(), temp) == NULL) {
            imread_raise(FileSystemError,
                "FATAL internal error raised during path::make_absolute() call to ::realpath():",
             FF("\t%s (%d)", std::strerror(errno), errno),
                "In reference to path value:",
             FF("\t%s", c_str()));
        }
        return path(temp);
    }
    
    bool path::compare_debug(const path &other) const {
        char raw_self[PATH_MAX],
             raw_other[PATH_MAX];
        if (::realpath(c_str(), raw_self)  == NULL) {
            imread_raise(FileSystemError,
                "FATAL internal error raised during path::compare_debug() call to ::realpath():",
             FF("\t%s (%d)", std::strerror(errno), errno),
                "In reference to SELF path value:",
                str());
        }
        if (::realpath(other.c_str(), raw_other) == NULL) {
            imread_raise(FileSystemError,
                "FATAL internal error raised during path::compare_debug() call to ::realpath():",
             FF("\t%s (%d)", std::strerror(errno), errno),
                "In reference to OTHER path value:",
                other.str());
        }
        WTF("COMPARING PATHS:",
            str(), other.str(),
            "AS RAW STRINGS:",
            raw_self, raw_other);
        return bool(std::strcmp(raw_self, raw_other) == 0);
    }
    
    bool path::compare_lexical(const path &other) const {
        char raw_self[PATH_MAX],
             raw_other[PATH_MAX];
        if (::realpath(c_str(),       raw_self)  == NULL) { return false; }
        if (::realpath(other.c_str(), raw_other) == NULL) { return false; }
        return bool(std::strcmp(raw_self, raw_other) == 0);
    }
    
    bool path::compare(const path &other) const {
        return bool(hash() == other.hash());
    }
    
    std::vector<path> path::list(bool full_paths) const {
        /// list all files
        if (!is_directory()) {
            imread_raise(FileSystemError,
                "Can't list files from a non-directory:", str());
        }
        path abspath = make_absolute();
        std::vector<path> out;
        {
            directory d = detail::ddopen(abspath.str());
            if (!d.get()) {
                imread_raise(FileSystemError,
                    "Internal error in opendir():", strerror(errno),
                    "For path:", str());
            }
            detail::dirent_t *entp;
            while ((entp = ::readdir(d.get())) != NULL) {
                if (std::strncmp(entp->d_name, ".", 1) == 0)   { continue; }
                if (std::strncmp(entp->d_name, "..", 2) == 0)  { continue; }
                /// ... it's either a directory, a regular file, or a symbolic link
                switch (entp->d_type) {
                    case DT_DIR:
                    case DT_REG:
                    case DT_LNK:
                        out.push_back(full_paths ? abspath/entp->d_name : path(entp->d_name));
                    default:
                        continue;
                }
            }
        } /// scope exit for d
        return out;
    }
    
    static const int glob_pattern_flags = GLOB_ERR | GLOB_NOSORT | GLOB_DOOFFS;
    
    std::vector<path> path::list(const char *pattern, bool full_paths) const {
        /// list files with glob
        if (!pattern) {
            imread_raise(FileSystemError,
                "Called path::list() with false-y pattern pointer on path:", str());
        }
        if (!is_directory()) {
            imread_raise(FileSystemError,
                "Bad call to path::list() from a non-directory:", str());
        }
        path abspath = make_absolute();
        glob_t g = {0};
        {
            filesystem::switchdir s(abspath);
            ::glob(pattern, glob_pattern_flags, NULL, &g);
        }
        std::vector<path> out;
        for (std::size_t idx = 0; idx != g.gl_pathc; ++idx) {
            out.push_back(full_paths ? abspath/g.gl_pathv[idx] : path(g.gl_pathv[idx]));
        }
        ::globfree(&g);
        return out;
    }
    
    std::vector<path> path::list(const std::string &pattern, bool full_paths) const {
        return list(pattern.c_str());
    }
    
    static const int regex_flags        = std::regex::extended;
    static const int regex_flags_icase  = std::regex::extended | std::regex::icase;
    
    std::vector<path> path::list(const std::regex &pattern, bool case_sensitive, bool full_paths) const {
        /// list files with regex object
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
    
    bool path::exists() const {
        detail::stat_t sb;
        return ::stat(c_str(), &sb) == 0;
    }
    
    bool path::is_file() const {
        detail::stat_t sb;
        if (::stat(c_str(), &sb)) { return false; }
        return S_ISREG(sb.st_mode);
    }
    
    bool path::is_directory() const {
        detail::stat_t sb;
        if (::stat(c_str(), &sb)) { return false; }
        return S_ISDIR(sb.st_mode);
    }
    
    bool path::remove() const {
        if (is_file())      { return bool(::unlink(make_absolute().c_str()) != -1); }
        if (is_directory()) { return bool(::rmdir(make_absolute().c_str()) != -1); }
        return false;
    }
    
    path path::getcwd() {
        char temp[PATH_MAX];
        if (::getcwd(temp, PATH_MAX) == NULL) {
            imread_raise(FileSystemError,
                "Internal error in getcwd():", strerror(errno));
        }
        return path(temp);
    }
    
    void path::swap(path& other) noexcept {
        using std::swap;
        m_path.swap(other.m_path);
        swap(m_absolute, other.m_absolute);
    }
    
    /// define static mutex, as declared in switchdir struct:
    std::mutex switchdir::mute;
    
} /* namespace filesystem */

namespace std {
    
    template <>
    void swap(filesystem::path& p0, filesystem::path& p1) {
        p0.swap(p1);
    }
    
}; /* namespace std */
