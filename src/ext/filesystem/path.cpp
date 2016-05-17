/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <sys/stat.h>
#include <sys/types.h>
#include <pwd.h>
#include <glob.h>
#include <fcntl.h>
#include <dlfcn.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#if defined(__APPLE__) || defined(__FreeBSD__)
#include <copyfile.h>
#else
#include <sys/sendfile.h>
#endif

#include <cerrno>
#include <cstdlib>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <type_traits>

#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/rehash.hh>

namespace filesystem {
    
    namespace detail {
        
        using stat_t = struct stat;
        using passwd_t = struct passwd;
        using rehasher_t = hash::rehasher<std::string>;
        // using inode_t = std::make_signed_t<::ino_t>;
        using dlinfo_t = ::Dl_info;
        
        char const* tmpdir() noexcept {
            /// cribbed/tweaked from boost
            char const* dirname;
            dirname = std::getenv("TMPDIR");
            if (NULL == dirname) { dirname = std::getenv("TMP"); }
            if (NULL == dirname) { dirname = std::getenv("TEMP"); }
            if (NULL == dirname) { dirname = "/tmp"; }
            return dirname;
        }
        
        char const* userdir() noexcept {
            char const* dirname;
            dirname = std::getenv("HOME");
            if (NULL == dirname) {
                passwd_t* pw = ::getpwuid(::geteuid());
                std::string dn(pw->pw_dir);
                dirname = dn.c_str();
            }
            return dirname;
        }
        
        char const* syspaths() noexcept {
            char const* syspaths;
            syspaths = std::getenv("PATH");
            if (NULL == syspaths) { syspaths = "/bin:/usr/bin"; }
            return syspaths;
        }
        
        std::string execpath() noexcept {
            char pbuf[PATH_MAX] = { 0 };
            uint32_t size = sizeof(pbuf);
            #ifdef __APPLE__
                if (_NSGetExecutablePath(pbuf, &size) != 0) { return ""; }
                ssize_t res = size;
            #else
                ssize_t res = ::readlink("/proc/self/exe", pbuf, size);
                if (res == 0 || res == sizeof(pbuf)) { return "" }
            #endif
            return std::string(pbuf, res);
        }
        
        ssize_t copyfile(char const* source, char const* destination) {
            /// Copy a file from source to destination
            /// Adapted from http://stackoverflow.com/a/2180157/298171
            int input, output; /// file descriptors
            
            if ((input  = ::open(source,      O_RDONLY)) == -1) { return -1; }
            if ((output = ::open(destination, O_RDWR | O_CREAT, 0644)) == -1) {
                ::close(input);
                return -1;
            }
            
            #if defined(__APPLE__) || defined(__FreeBSD__)
                /// fcopyfile() works on FreeBSD and OS X 10.5+ 
                ssize_t result = ::fcopyfile(input, output, 0, COPYFILE_ALL);
            #else
                /// sendfile() will work with non-socket output
                /// -- i.e. regular files -- on Linux 2.6.33+
                off_t bytescopied = 0;
                stat_t fileinfo = { 0 };
                ::fstat(input, &fileinfo);
                ssize_t result = ::sendfile(output, input, &bytescopied, fileinfo.st_size);
            #endif
            
            ::close(input);
            ::close(output);
            
            return result;
        }
        
    } /* namespace detail */
    
    static const std::regex::flag_type regex_flags        = std::regex::extended;
    static const std::regex::flag_type regex_flags_icase  = std::regex::extended | std::regex::icase;
    
    constexpr path::character_type path::sep;
    constexpr path::character_type path::extsep;
    
    path::path()
        :m_type(native_path)
        ,m_absolute(false)
        {}
    
    path::path(path const& p)
        :m_type(native_path)
        ,m_path(p.m_path)
        ,m_absolute(p.m_absolute)
        {}
    
    path::path(path&& p) noexcept
        :m_type(native_path)
        ,m_path(std::move(p.m_path))
        ,m_absolute(p.m_absolute)
        {}
    
    path::path(char* st)              { set(st); }
    path::path(char const* st)        { set(st); }
    path::path(std::string const& st) { set(st); }
    
    path::path(int descriptor) {
        char fdpath[PATH_MAX];
        int return_value = ::fcntl(descriptor, F_GETPATH, fdpath);
        if (return_value != -1) {
            set(fdpath);
        } else {
            imread_raise(FileSystemError,
                "Internal error in ::fnctl(descriptor, F_GETPATH, fdpath)",
                "where fdpath = ", fdpath, std::strerror(errno));
        }
    }
    
    path::path(const void* address) {
        detail::dlinfo_t dlinfo;
        if (::dladdr(address, &dlinfo)) {
            set(dlinfo.dli_fname);
        } else {
            imread_raise(FileSystemError,
                "Internal error in ::dlfcn(address, &dlinfo)",
                "where address = ", (long)address, std::strerror(errno));
        }
    }
    
    path::path(detail::stringvec_t const& vec, bool absolute)
        :m_absolute(absolute), m_type(native_path)
        ,m_path(vec)
        {}
    
    path::path(detail::stringlist_t list)
        :m_absolute(false), m_type(native_path)
        ,m_path(list)
        {}
    
    std::size_t path::size() const { return static_cast<std::size_t>(m_path.size()); }
    bool path::is_absolute() const { return m_absolute; }
    bool path::empty() const       { return m_path.empty(); }
    
    detail::inode_t path::inode() const  {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return detail::null_inode_v; }
        return static_cast<detail::inode_t>(sb.st_ino);
    }
    
    #define REGEX_FLAGS case_sensitive ? regex_flags_icase : regex_flags
    
    bool path::match(std::regex const& pattern, bool case_sensitive) const {
        return std::regex_match(str(), pattern);
    }
    
    bool path::search(std::regex const& pattern, bool case_sensitive) const {
        return std::regex_search(str(), pattern);
    }
    
    path path::replace(std::regex const& pattern, char const* replacement, bool case_sensitive) const {
        return path(std::regex_replace(str(), pattern, replacement));
    }
    path path::replace(std::regex const& pattern, std::string const& replacement, bool case_sensitive) const {
        return path(std::regex_replace(str(), pattern, replacement));
    }
    
    #undef REGEX_FLAGS
    
    path path::make_absolute() const {
        if (m_absolute) { return path(*this); }
        char temp[PATH_MAX];
        if (::realpath(c_str(), temp) == NULL) {
            imread_raise(FileSystemError,
                "In reference to path value:", c_str(),
                "FATAL internal error raised during path::make_absolute() call to ::realpath():",
                std::strerror(errno));
        }
        return path(temp);
    }
    
    path path::expand_user() const {
        const std::regex re("^~", regex_flags);
        if (m_path.empty()) { return path(); }
        if (m_path[0].substr(0, 1) != "~") { return path(*this); }
        return replace(re, detail::userdir());
    }
    
    bool path::compare_debug(path const& other) const {
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
    
    bool path::compare_lexical(path const& other) const {
        char raw_self[PATH_MAX],
             raw_other[PATH_MAX];
        if (::realpath(c_str(),       raw_self)  == NULL) { return false; }
        if (::realpath(other.c_str(), raw_other) == NULL) { return false; }
        return bool(std::strcmp(raw_self, raw_other) == 0);
    }
    
    bool path::compare(path const& other) const noexcept {
        return bool(hash() == other.hash());
    }
    
    detail::pathvec_t path::list(bool full_paths) const {
        /// list all files
        if (!is_directory()) {
            imread_raise(FileSystemError,
                "Can't list files from a non-directory:", str());
        }
        path abspath = make_absolute();
        detail::pathvec_t out;
        {
            directory d = detail::ddopen(abspath.str());
            if (!d.get()) {
                imread_raise(FileSystemError,
                    "Internal error in opendir():", strerror(errno),
                    "For path:", str());
            }
            detail::dirent_t* entp;
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
    
    detail::vector_pair_t path::list(detail::list_separate_t tag, bool full_paths) const {
        detail::stringvec_t directories;
        detail::stringvec_t files;
        {
            directory d = detail::ddopen(make_absolute().str());
            if (!d.get()) {
                imread_raise(FileSystemError,
                    "Internal error in opendir():", strerror(errno),
                    "For path:", str());
            }
            detail::dirent_t* entp;
            while ((entp = ::readdir(d.get())) != NULL) {
                if (std::strncmp(entp->d_name, ".", 1) == 0)   { continue; }
                if (std::strncmp(entp->d_name, "..", 2) == 0)  { continue; }
                /// ... it's either a directory, a regular file, or a symbolic link
                std::string t(entp->d_name);
                switch (entp->d_type) {
                    case DT_DIR:
                        directories.push_back(std::move(t));
                        continue;
                    case DT_REG:
                    case DT_LNK:
                        files.push_back(std::move(t));
                        continue;
                    default:
                        continue;
                }
            }
        } /// scope exit for d
        return std::make_pair(
            std::move(directories),
            std::move(files));
    }
    
    static const int glob_pattern_flags = GLOB_ERR | GLOB_NOSORT | GLOB_DOOFFS;
    
    detail::pathvec_t path::list(char const* pattern, bool full_paths) const {
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
        detail::pathvec_t out;
        for (std::size_t idx = 0; idx != g.gl_pathc; ++idx) {
            out.push_back(full_paths ? abspath/g.gl_pathv[idx] : path(g.gl_pathv[idx]));
        }
        ::globfree(&g);
        return out;
    }
    
    detail::pathvec_t path::list(std::string const& pattern, bool full_paths) const {
        return list(pattern.c_str(), full_paths);
    }
    
    detail::pathvec_t path::list(std::regex const& pattern, bool case_sensitive, bool full_paths) const {
        /// list files with regex object
        detail::pathvec_t unfiltered = list(full_paths);
        detail::pathvec_t out;
        if (unfiltered.size() == 0) { return out; }
        std::copy_if(unfiltered.begin(), unfiltered.end(),
                     std::back_inserter(out),
                     [&](path const& p) {
            /// keep those paths that match the pattern
            return p.search(pattern, case_sensitive);
        });
        return out;
    }
    
    // using walk_visitor_t = std::function<void(const path&,       /// root path
    //                                      detail::stringvec_t&,   /// directories
    //                                      detail::stringvec_t&)>; /// files
    
    void path::walk(detail::walk_visitor_t&& walk_visitor) const {
        if (!is_directory()) {
            imread_raise(FileSystemError,
                "Can't call path::walk() on a non-directory:", str());
        }
        
        /// list with tag dispatch for separate return vectors
        const detail::list_separate_t tag{};
        const path abspath = make_absolute();
        detail::vector_pair_t vector_pair = abspath.list(tag);
        
        /// separate out files and directories
        detail::stringvec_t directories = std::move(vector_pair.first);
        detail::stringvec_t files = std::move(vector_pair.second);
        
        /// walk_visitor() may modify `directories`
        std::forward<detail::walk_visitor_t>(walk_visitor)(abspath, directories, files);
        
        /// recursively walk into subdirs
        if (directories.empty()) { return; }
        std::for_each(directories.begin(), directories.end(),
                      [&](std::string const& subdir) {
            abspath.join(subdir).walk(
                std::forward<detail::walk_visitor_t>(walk_visitor));
        });
    }
    
    bool path::exists() const {
        detail::stat_t sb;
        return ::lstat(c_str(), &sb) == 0;
    }
    
    bool path::is_file() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return S_ISREG(sb.st_mode);
    }
    
    bool path::is_link() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return S_ISLNK(sb.st_mode);
    }
    
    bool path::is_directory() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return S_ISDIR(sb.st_mode);
    }
    
    bool path::is_file_or_link() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return bool(S_ISREG(sb.st_mode)) || bool(S_ISLNK(sb.st_mode));
    }
    
    bool path::remove() const {
        if (is_file_or_link()) { return bool(::unlink(make_absolute().c_str()) != -1); }
        if (is_directory())    { return bool(::rmdir(make_absolute().c_str()) != -1); }
        return false;
    }
    
    static const mode_t mkdir_flags = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    
    bool path::makedir() const {
        if (empty() || exists()) { return false; }
        return bool(::mkdir(c_str(), mkdir_flags) != -1);
    }
    
    std::string path::basename() const {
        if (empty()) { return ""; }
        return m_path.back();
    }
    
    bool path::rename(path const& newpath) {
        if (!exists() || newpath.exists()) { return false; }
        bool status = ::rename(c_str(), newpath.c_str()) != -1;
        if (status) { set(newpath.make_absolute().str()); }
        return status;
    }
    bool path::rename(char const* newpath)        { return rename(path(newpath)); }
    bool path::rename(std::string const& newpath) { return rename(path(newpath)); }
    
    path path::duplicate(path const& newpath) {
        path out;
        if (!exists() || newpath.exists()) { return out; }
        bool status = detail::copyfile(c_str(), newpath.c_str()) != -1;
        if (status) { out = path::absolute(newpath); }
        return out;
    }
    path path::duplicate(char const* newpath)        { return duplicate(path(newpath)); }
    path path::duplicate(std::string const& newpath) { return duplicate(path(newpath)); }
    
    std::string path::extension() const {
        if (empty()) { return ""; }
        std::string const& last = m_path.back();
        size_type pos = last.find_last_of(extsep);
        if (pos == std::string::npos) { return ""; }
        return last.substr(pos+1);
    }
    
    path path::parent() const {
        path result;
        result.m_absolute = m_absolute;
        
        if (m_path.empty()) {
            if (!m_absolute) {
                result.m_path.push_back("..");
            } else {
                imread_raise(FileSystemError,
                    "path::parent() can't get the parent of an empty absolute path");
            }
        } else {
            size_type idx = 0,
                      until = m_path.size() - 1;
            for (; idx < until; ++idx) {
                result.m_path.push_back(m_path[idx]);
            }
        }
        return result;
    }
    path path::dirname() const { return parent(); }
    
    path path::join(path const& other) const {
        if (other.m_absolute) {
            imread_raise(FileSystemError,
                "path::join() expects a relative-path RHS");
        }
        
        path result(m_path, m_absolute);
        size_type idx = 0,
                  max = other.m_path.size();
        
        for (; idx < max; ++idx) {
            result.m_path.push_back(other.m_path[idx]);
        }
        return result;
    }
    
    path path::append(std::string const& appendix) const {
        path out(m_path, m_absolute);
        out.m_path.back().append(appendix);
        return out;
    }
    
    std::string path::str() const {
        std::string out = "";
        if (m_absolute) { out += sep; }
        size_type idx = 0,
                  siz = m_path.size();
        for (; idx < siz; ++idx) {
            out += m_path[idx];
            if (idx + 1 < siz) { out += sep; }
        }
        return out;
    }
    
    char const* path::c_str() const {
        return str().c_str();
    }
    
    path path::getcwd() {
        char temp[PATH_MAX];
        if (::getcwd(temp, PATH_MAX) == NULL) {
            imread_raise(FileSystemError,
                "Internal error in getcwd():",
                std::strerror(errno));
        }
        return path(temp);
    }
    
    path path::cwd()                { return path::getcwd(); }
    path path::gettmp()             { return path(detail::tmpdir()); }
    path path::tmp()                { return path(detail::tmpdir()); }
    path path::home()               { return path(detail::userdir()); }
    path path::user()               { return path(detail::userdir()); }
    path path::executable()         { return path(detail::execpath()); }
    
    detail::stringvec_t path::system() {
        return tokenize(detail::syspaths(), detail::posix_pathvar_separator);
    }
    
    void path::set(std::string const& str) {
        m_type = native_path;
        m_path = tokenize(str, sep);
        m_absolute = !str.empty() && str[0] == sep;
    }
    
    /// calculate the hash value for the path
    std::size_t path::hash() const noexcept {
        std::size_t seed = static_cast<std::size_t>(m_absolute);
        return std::accumulate(m_path.begin(), m_path.end(),
                               seed, detail::rehasher_t());
    }
    
    void path::swap(path& other) noexcept {
        using std::swap;
        swap(m_path, other.m_path);
        swap(m_absolute, other.m_absolute);
    }
    
    /// path component vector
    detail::stringvec_t path::components() const {
        detail::stringvec_t out;
        std::copy(m_path.begin(), m_path.end(),
                  std::back_inserter(out));
        return out;
    }
    
    detail::stringvec_t path::tokenize(std::string const& source,
                                       path::character_type const delim) {
        detail::stringvec_t tokens;
        path::size_type lastPos = 0,
                        pos = source.find_first_of(delim, lastPos);
        
        while (lastPos != std::string::npos) {
            if (pos != lastPos) {
                tokens.push_back(source.substr(lastPos, pos - lastPos));
            }
            lastPos = pos;
            if (lastPos == std::string::npos || lastPos + 1 == source.length()) { break; }
            pos = source.find_first_of(delim, ++lastPos);
        }
        
        return tokens;
    }
    
    
    /// define static mutex,
    /// as declared in switchdir struct:
    std::mutex switchdir::mute;
    
    /// define static recursive mutex and global path stack,
    /// as declared in workingdir struct:
    std::recursive_mutex workingdir::mute;
    std::stack<path> workingdir::dstack;
    const path workingdir::empty = path();
    
} /* namespace filesystem */

namespace std {
    
    template <>
    void swap(filesystem::path& p0, filesystem::path& p1) noexcept {
        p0.swap(p1);
    }
    
}; /* namespace std */
