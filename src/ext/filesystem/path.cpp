/// Copyright 2014-2018 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <utime.h>
#include <pwd.h>
#include <glob.h>
#include <wordexp.h>
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

#include <cstdlib>
#include <cerrno>
#include <thread>
#include <numeric>
#include <algorithm>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/attributes.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/nowait.h>
#include <libimread/ext/filesystem/opaques.h>
#include <libimread/ext/pystring.hh>
#include <libimread/errors.hh>
#include <libimread/rehash.hh>

using im::byte;
using namespace std::chrono_literals;

namespace filesystem {
    
    namespace detail {
        
        using stat_t = struct stat;
        using passwd_t = struct passwd;
        using rehasher_t = hash::rehasher<std::string>;
        using dlinfo_t = ::Dl_info;
        using scandir_f = std::add_pointer_t<int(detail::dirent_t*)>;
        using kinfo_proc_t = struct kinfo_proc;
        using timeval_t = struct timeval;
        
        std::string tmpdir() noexcept {
            /// Returns a std::string containing the temporary directory,
            /// using std::getenv() and some guesswork -- originally cribbed from boost
            char const* dirname = nullptr;
            dirname = std::getenv("TMPDIR");
            if (nullptr == dirname) { dirname = std::getenv("TMP"); }
            if (nullptr == dirname) { dirname = std::getenv("TEMP"); }
            if (nullptr == dirname) { dirname = "/tmp"; }
            return dirname;
        }
        
        std::string username() noexcept {
            /// Returns a std::string containing the users’ login name,
            /// using either std::getenv() or, barring that, POSIX ::getlogin().
            char const* username = nullptr;
            username = std::getenv("USER");
            if (nullptr == username) { username = std::getenv("LOGNAME"); }
            if (nullptr == username) { username = ::getlogin(); }
            if (nullptr == username) { username = "nobody"; }
            return username;
        }
        
        std::string userdir() noexcept {
            /// Returns a std::string containing the users’ home directory,
            /// using std::getenv(), POSIX passwd-related calls, and some guesswork:
            char const* dirname = nullptr;
            passwd_t* pw = nullptr;
            dirname = std::getenv("HOME");
            if (nullptr == dirname) {
                std::string user = filesystem::detail::username();
                pw = ::getpwnam(user.c_str());
                if (pw && pw->pw_dir) {
                    std::string dn(pw->pw_dir);
                    ::endpwent(); /// release password-database-related resources
                    return dn;
                }
            }
            if (nullptr == dirname) {
                pw = ::getpwuid(::geteuid());
                if (pw && pw->pw_dir) {
                    std::string dn(pw->pw_dir);
                    ::endpwent(); /// release password-database-related resources
                    return dn;
                }
            }
            if (nullptr == dirname) { dirname = "/dev/null"; }
            return dirname;
        }
        
        std::string syspaths() noexcept {
            /// Returns a std::string containing the system paths (aka the $PATH variable),
            /// using std::getenv() with some sensible defaults:
            char const* paths = nullptr;
            paths = std::getenv("PATH");
            if (nullptr == paths) { paths = "/bin:/usr/bin"; }
            return paths;
        }
        
        std::string execpath() noexcept {
            /// Returns a std::string with the path to the calling executable,
            /// using either _NSGetExecutablePath() or /proc/self/exe:
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
        
        timepoint_t execstarttime() noexcept {
            /// Returns the calling process’ startup time as a UNIX timestamp,
            /// using getpid(), sysctl(), `struct kinfo_proc`, and `struct timeval` --
            /// Adapted from https://github.com/ibireme/tmp/blob/master/snippet/app_startup.m:
            int mib[4] = {
                CTL_KERN,
                KERN_PROC,
                KERN_PROC_PID,
                ::getpid() /// current pid
            };
            
            kinfo_proc_t kinfo = { 0 };
            std::size_t mib_size = sizeof(mib) / sizeof(*mib);
            std::size_t kinfo_size = sizeof(kinfo);
            std::size_t zero_size = 0;
            
            int result = ::sysctl(mib,      mib_size,
                                 &kinfo,   &kinfo_size,
                                  nullptr,  zero_size);
            
            if (result != 0) { return clock_t::from_time_t(0); } /// ERROR!
            
            timeval_t* tv = &kinfo.kp_proc.p_un.__p_starttime;
            uint64_t ts = (tv->tv_sec * static_cast<uint64_t>(1000)) + (tv->tv_usec / 1000);
            return clock_t::from_time_t(ts);
        }
        
        static constexpr int    copyfile_source_flags           = O_RDONLY | O_CLOEXEC;
        static constexpr int    copyfile_destination_flags      = O_RDWR   | O_CLOEXEC | O_CREAT;
        static constexpr mode_t copyfile_destination_mode       = 0644;
        
        #if defined(__APPLE__) || defined(__FreeBSD__)
        static constexpr copyfile_flags_t copy_with_xattrs      = COPYFILE_SECURITY | COPYFILE_XATTR | COPYFILE_DATA;
        static constexpr copyfile_flags_t copy_without_xattrs   = COPYFILE_SECURITY | COPYFILE_DATA;
        #endif
        
        ssize_t copyfile(char const* source, char const* destination, bool copy_attributes = true) {
            /// Copy a file from source to destination
            /// Adapted from http://stackoverflow.com/a/2180157/298171
            int input, output; /// file descriptors
            
            if ((input  = ::open(source,
                                 copyfile_source_flags)) == -1)     {   return -1; }
            if ((output = ::open(destination,
                                 copyfile_destination_flags,
                                 copyfile_destination_mode)) == -1) { ::close(input);
                                                                        return -1; }
            
            #if defined(__APPLE__) || defined(__FreeBSD__)
                /// fcopyfile() works on FreeBSD and OS X 10.5+
                ssize_t result = ::fcopyfile(input,
                                             output,
                                             nullptr,
                                             copy_attributes ? copy_with_xattrs
                                                             : copy_without_xattrs);
                copy_attributes = false; /// don’t manually copy xattrs
            
            #else
                /// sendfile() will work with non-socket output
                /// -- i.e. regular files -- on Linux 2.6.33+
                off_t offset = 0;
                stat_t source_info = { 0 };
                ::fstat(input, &source_info);
                ssize_t result = ::sendfile(output,
                                            input,
                                           &offset,
                                            source_info.st_size);
            
            #endif
            
            if (copy_attributes) {
                if (attribute::fdcount(input) > 0) {
                    if (::access(destination, R_OK | W_OK) != -1) {
                        for (std::string const& name : attribute::fdlist(input)) {
                            attribute::fdset(output, name,
                            attribute::fdget(input, name));
                        }
                    }
                }
            }
            
            ::close(input);
            ::close(output);
            
            return result;
        }
        
        static constexpr std::regex::flag_type regex_flags          = std::regex::extended;
        static constexpr std::regex::flag_type regex_flags_icase    = std::regex::extended | std::regex::icase;
        static constexpr int glob_pattern_flags                     = GLOB_ERR | GLOB_NOSORT | GLOB_DOOFFS;
        static constexpr int wordexp_pattern_flags                  = WRDE_NOCMD | WRDE_DOOFFS;
        static constexpr mode_t mkfifo_flags                        = S_IRUSR  | S_IWUSR | S_IRGRP  | S_IWGRP   | S_IROTH | S_IWOTH | S_IFIFO;
        static constexpr mode_t mkdir_flags                         = S_IRWXU  | S_IRWXG | S_IROTH  | S_IXOTH;
        static constexpr int touch_open_flags                       = O_WRONLY | O_CREAT | O_NOCTTY | O_NONBLOCK;
        static constexpr mode_t touch_open_mask                     = 0666;
        
        /// Re-implementation of the logic behind the “touch” UNIX command-line tool,
        /// as elucidated by someone named Chris, per this discursive analysis:
        ///     http://chris-sharpe.blogspot.com/2013/05/better-than-systemtouch.html
        /// My rewrite plugs the descriptor leak, and also avoids playing the race card
        /// with regard to our target path, qua the two system calls made.
        
        bool touch(char const* target) noexcept {
            /// Sanity-check `target`:
            if (!target) { return false; }
            
            /// Open and obtain a descriptor for our target path -- using flags specifying
            /// that the file is to be created in the event of its nonexistence:
            int descriptor = ::open(target,
                                    filesystem::detail::touch_open_flags,
                                    filesystem::detail::touch_open_mask);
            
            /// Bail if ::open() choked:
            if (descriptor < 0) { return false; }
            
            /// Update the timestamps for our freshly-opened file, through its descriptor --
            /// A call to ::futimens() with the second arg (of type `const struct timespec*`)
            /// value of `nullptr` means “set timestamps to whatever the system clocks’ value
            /// is for right now”:
            int status = ::futimens(descriptor, nullptr);
            
            /// Close out the descriptor:
            ::close(descriptor);
            
            /// Return, per the inverse of the status result of the ::futimens() call:
            return !bool(status);
        }
        
        static const std::string extsepstring(1, path::extsep);
        static const std::string sepstring(1, path::sep);
        static const std::string nulstring("");
        static const std::regex  user_directory_re("^~" + sepstring, regex_flags);
        static const std::regex  user_directory_prefix_re("^~([A-Za-z][A-Za-z0-9_]*)", regex_flags);
        
    } /* namespace detail */
    
    constexpr path::character_type path::sep;
    constexpr path::character_type path::extsep;
    constexpr path::character_type path::pathsep;
    
    path::path() {}
    path::path(bool abs)
        :m_absolute(abs)
        {}
    
    path::path(path const& p)
        :m_absolute(p.m_absolute)
        ,m_path(p.m_path)
        {}
    
    path::path(path&& p) noexcept
        :m_absolute(p.m_absolute)
        ,m_path(std::move(p.m_path))
        {}
    
    path::path(char* st)              { set(st); }
    path::path(char const* st)        { set(st); }
    path::path(std::string const& st) { set(st); }
    
    path::path(int descriptor) {
        /// q.v. https://github.com/textmate/textmate/blob/master/Frameworks/io/src/path.cc#L587
        ///      ... w/r/t why we do ::fcntl() twice:
        for (path::size_type idx = 0; idx < 100; ++idx) {
            char fdpath[PATH_MAX];
            bool result = (::fcntl(descriptor, F_GETPATH, fdpath) == 0);
            #ifdef __APPLE__
            /// A workaround, allegedly, for <rdar://6149247>
            result &=     (::fcntl(descriptor, F_GETPATH, fdpath) == 0);
            #endif
            result &=     (::access(fdpath, F_OK) == 0);
            if (result) {
                set(fdpath);
                return;
            }
            std::this_thread::sleep_for(10us);
        }
    }
    
    path::path(const void* address) {
        detail::dlinfo_t dlinfo;
        if (::dladdr(address, &dlinfo)) {
            set(dlinfo.dli_fname);
        } else {
            imread_raise(FileSystemError,
                "Internal error in ::dladdr(address, &dlinfo)",
                "where address = ", reinterpret_cast<long>(address),
                std::strerror(errno));
        }
    }
    
    path::path(detail::stringvec_t const& vec, bool absolute)
        :m_absolute(absolute)
        ,m_path(vec)
        {}
    
    path::path(detail::stringvec_t&& vec, bool absolute) noexcept
        :m_absolute(absolute)
        ,m_path(std::move(vec))
        {}
    
    path::path(detail::stringlist_t list)
        :m_path(list)
        {}
    
    path::~path() {}
    
    path::size_type path::size() const { return static_cast<path::size_type>(m_path.size()); }
    bool path::is_absolute() const { return m_absolute; }
    bool path::empty() const       { return m_path.empty(); }
    
    detail::inode_t path::inode() const  {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return detail::null_inode_v; }
        return { static_cast<dev_t>(sb.st_dev),
                 static_cast<ino_t>(sb.st_ino) };
    }
    
    path::size_type path::filesize() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return 0; }
        return sb.st_size * sizeof(byte);
    }
    
    path::size_type path::hardlinkcount() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return 0; }
        return sb.st_nlink;
    }
    
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
    
    path path::make_absolute() const {
        if (m_absolute) { return path(*this); }
        char temp[PATH_MAX];
        if (::realpath(c_str(), temp) == nullptr) {
            imread_raise(FileSystemError,
                "In reference to path value:", str(),
                "FATAL internal error in path::make_absolute() call to ::realpath():",
                std::strerror(errno));
        }
        return path(temp);
    }
    
    path path::make_real() const {
        /// same as path::make_absolute(), minus the m_absolute precheck:
        char temp[PATH_MAX];
        if (::realpath(c_str(), temp) == nullptr) {
            imread_raise(FileSystemError,
                "In reference to path value:", str(),
                "FATAL internal error in path::make_real() call to ::realpath():",
                std::strerror(errno));
        }
        return path(temp);
    }
    
    path path::expand_user() const {
        if (m_path.empty()) { return path(); }
        if (m_path.front().substr(0, 1) != "~") { return path(*this); }
        return replace(detail::user_directory_re, detail::userdir());
    }
    
    bool path::compare_debug(path const& other) const {
        if (!exists() || !other.exists()) { return false; }
        char raw_self[PATH_MAX],
             raw_other[PATH_MAX];
        if (::realpath(c_str(), raw_self) == nullptr) {
            imread_raise(FileSystemError,
                "FATAL internal error in path::compare_debug() call to ::realpath():",
             FF("\t%s (%d)", std::strerror(errno), errno),
                "In reference to SELF path value:",
                str());
        }
        if (::realpath(other.c_str(), raw_other) == nullptr) {
            imread_raise(FileSystemError,
                "FATAL internal error in path::compare_debug() call to ::realpath():",
             FF("\t%s (%d)", std::strerror(errno), errno),
                "In reference to OTHER path value:",
                other.str());
        }
        return bool(std::strcmp(raw_self, raw_other) == 0);
    }
    
    bool path::compare_lexical(path const& other) const {
        if (!exists() || !other.exists()) { return false; }
        char raw_self[PATH_MAX],
             raw_other[PATH_MAX];
        if (::realpath(c_str(),       raw_self)  == nullptr) { return false; }
        if (::realpath(other.c_str(), raw_other) == nullptr) { return false; }
        return bool(std::strcmp(raw_self, raw_other) == 0);
    }
    
    bool path::compare_inodes(path const& other) const {
        detail::stat_t sb_lhs, sb_rhs;
        if (::lstat(c_str(),       &sb_lhs)) { return false; }
        if (::lstat(other.c_str(), &sb_rhs)) { return false; }
        return (sb_lhs.st_dev == sb_rhs.st_dev) &&
               (sb_lhs.st_ino == sb_rhs.st_ino);
    }
    
    bool path::compare(path const& other) const noexcept {
        return bool(hash() == other.hash());
    }
    
    detail::pathvec_t path::list(bool full_paths) const {
        /// list all files
        detail::pathvec_t out;
        if (!is_listable()) { return out; }
        {
            detail::nowait_t nowait;
            path abspath = make_absolute();
            directory d = detail::ddopen(abspath.str());
            if (!d.get()) {
                imread_raise(FileSystemError,
                    "Internal error in opendir():", std::strerror(errno),
                    "For path:", str());
            }
            detail::dirent_t* entp;
            if (full_paths) {
                /// concatenate new paths with `abspath`
                while ((entp = ::readdir(d.get())) != nullptr) {
                    /// ... it's either a directory, a regular file, or a symbolic link
                    switch (entp->d_type) {
                        case DT_DIR:
                            if (std::strcmp(entp->d_name, ".") == 0)   { continue; }
                            if (std::strcmp(entp->d_name, "..") == 0)  { continue; }
                        case DT_REG:
                        case DT_LNK:
                        case DT_UNKNOWN:
                            out.emplace_back(abspath/entp->d_name);
                        default:
                            continue;
                    }
                }
            } else {
                /// make new relative paths
                while ((entp = ::readdir(d.get())) != nullptr) {
                    /// ... it's either a directory, a regular file, or a symbolic link
                    switch (entp->d_type) {
                        case DT_DIR:
                            if (std::strcmp(entp->d_name, ".") == 0)   { continue; }
                            if (std::strcmp(entp->d_name, "..") == 0)  { continue; }
                        case DT_REG:
                        case DT_LNK:
                        case DT_UNKNOWN:
                            out.emplace_back(entp->d_name);
                        default:
                            continue;
                    }
                }
            }
        } /// scope exit for d and nowait
        return out;
    }
    
    detail::vector_pair_t path::list(detail::list_separate_t tag, bool full_paths) const {
        detail::stringvec_t directories;
        detail::stringvec_t files;
        {
            detail::nowait_t nowait;
            directory d = detail::ddopen(make_absolute().str());
            if (!d.get()) {
                imread_raise(FileSystemError,
                    "Internal error in opendir():", std::strerror(errno),
                    "For path:", str());
            }
            detail::dirent_t* entp;
            while ((entp = ::readdir(d.get())) != nullptr) {
                /// ... it's either a directory, a regular file, or a symbolic link
                switch (entp->d_type) {
                    case DT_DIR:
                        if (std::strcmp(entp->d_name, ".") == 0)   { continue; }
                        if (std::strcmp(entp->d_name, "..") == 0)  { continue; }
                        directories.emplace_back(entp->d_name);
                        continue;
                    case DT_REG:
                    case DT_LNK:
                    case DT_UNKNOWN:
                        files.emplace_back(entp->d_name);
                        continue;
                    default:
                        continue;
                }
            }
        } /// scope exit for d and nowait
        return std::make_pair(
            std::move(directories),
            std::move(files));
    }
    
    detail::pathvec_t path::list(char const* pattern, bool full_paths) const {
        /// list files with glob
        detail::pathvec_t out;
        if (!is_listable()) { return out; }
        path abspath = make_absolute();
        ::glob_t glob_s = { 0 };
        {
            detail::nowait_t nowait;
            filesystem::switchdir s(abspath);
            ::glob(pattern, detail::glob_pattern_flags, nullptr, &glob_s);
        }
        path::size_type idx = 0,
                        max = glob_s.gl_pathc;
        out.reserve(max);
        if (full_paths) {
            for (; idx != max; ++idx) {
                out.emplace_back(abspath/glob_s.gl_pathv[idx]);
            }
        } else {
            for (; idx != max; ++idx) {
                out.emplace_back(glob_s.gl_pathv[idx]);
            }
        }
        ::globfree(&glob_s);
        return out;
    }
    
    detail::pathvec_t path::list(std::string const& pattern, bool full_paths) const {
        /// list files with wordexp
        detail::pathvec_t out;
        if (!is_listable()) { return out; }
        path abspath = make_absolute();
        ::wordexp_t word = { 0 };
        {
            detail::nowait_t nowait;
            filesystem::switchdir s(abspath);
            ::wordexp(pattern.c_str(), &word, detail::wordexp_pattern_flags);
        }
        path::size_type idx = 0,
                        max = word.we_wordc;
        out.reserve(max);
        if (full_paths) {
            for (; idx != max; ++idx) {
                out.emplace_back(abspath/word.we_wordv[idx]);
            }
        } else {
            for (; idx != max; ++idx) {
                out.emplace_back(word.we_wordv[idx]);
            }
        }
        ::wordfree(&word);
        return out;
    }
    
    detail::pathvec_t path::list_rx(std::regex const& pattern, bool full_paths,
                                                               bool case_sensitive) const {
        /// list files, filtered with a regex object
        detail::pathvec_t unfiltered = list(full_paths);
        detail::pathvec_t out;
        std::copy_if(unfiltered.begin(),
                     unfiltered.end(),
                     std::back_inserter(out),
                     [&pattern, &case_sensitive](path const& p) {
            /// keep any matching paths:
            return p.search(pattern, case_sensitive);
        });
        return out;
    }
    
    detail::pathvec_t path::list(std::regex const& pattern, bool full_paths) const {
        return list_rx(pattern, full_paths, true); /// <---- case_sensitive=true
    }
    
    detail::pathvec_t path::ilist(std::regex const& pattern, bool full_paths) const {
        return list_rx(pattern, full_paths, false); /// <---- case_sensitive=false
    }
    
    // using walk_visitor_t = std::function<void(const path&,       /// root path
    //                                      detail::stringvec_t&,   /// directories
    //                                      detail::stringvec_t&)>; /// files
    
    void path::walk(detail::walk_visitor_t&& walk_visitor) const {
        if (!is_listable()) { return; }
        
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
        std::for_each(directories.begin(),
                      directories.end(),
                  [&](std::string const& subdir) {
            abspath.join(subdir).walk(
                std::forward<detail::walk_visitor_t>(walk_visitor));
        });
    }
    
    bool path::operator==(path const& other) const { return bool(hash() == other.hash()); }
    bool path::operator!=(path const& other) const { return bool(hash() != other.hash()); }
    
    bool path::exists() const {
        return ::access(c_str(), F_OK) != -1;
    }
    
    bool path::is_readable() const {
        return ::access(c_str(), R_OK) != -1;
    }
    
    bool path::is_writable() const {
        return ::access(c_str(), W_OK) != -1;
    }
    
    bool path::is_executable() const {
        return (::access(c_str(), X_OK) != -1) && !is_directory();
    }
    
    bool path::is_readwritable() const {
        return ::access(c_str(), R_OK | W_OK) != -1;
    }
    
    bool path::is_runnable() const {
        return (::access(c_str(), R_OK | X_OK) != -1) && !is_directory();
    }
    
    bool path::is_listable() const {
        return (::access(c_str(), R_OK) != -1) && is_directory();
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
    
    bool path::is_block_device() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return S_ISBLK(sb.st_mode);
    }
    
    bool path::is_character_device() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return S_ISCHR(sb.st_mode);
    }
    
    bool path::is_pipe() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return S_ISFIFO(sb.st_mode);
    }
    
    bool path::is_file_or_link() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return false; }
        return bool(S_ISREG(sb.st_mode)) || bool(S_ISLNK(sb.st_mode));
    }
    
    long path::max_file_name_length() const {
        if (!is_directory()) { return -1L; }
        return ::pathconf(c_str(), _PC_NAME_MAX);
    }
    
    long path::max_relative_path_length() const {
        if (!is_directory()) { return -1L; }
        return ::pathconf(c_str(), _PC_PATH_MAX);
    }
    
    detail::time_triple_t path::timestamps() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return detail::time_triple_t{}; }
        return std::make_tuple(
            detail::clock_t::from_time_t(sb.st_atime),
            detail::clock_t::from_time_t(sb.st_mtime),
            detail::clock_t::from_time_t(sb.st_ctime));
    }
    
    detail::timepoint_t path::access_time() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return detail::timepoint_t{}; }
        return detail::clock_t::from_time_t(sb.st_atime);
    }
    
    detail::timepoint_t path::modify_time() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return detail::timepoint_t{}; }
        return detail::clock_t::from_time_t(sb.st_mtime);
    }
    
    detail::timepoint_t path::status_time() const {
        detail::stat_t sb;
        if (::lstat(c_str(), &sb)) { return detail::timepoint_t{}; }
        return detail::clock_t::from_time_t(sb.st_ctime);
    }
    
    bool path::update_timestamps() const {
        if (!exists()) return false;
        return ::utimes(c_str(), nullptr) != -1;
    }
    
    /// For `path::touch()`, we delegate to `detail::touch(…)` -- q.v. comment notes supra.
    bool path::touch() const {
        return detail::touch(c_str());
    }
    
    /// The path::touch() method’s return value doesn’t reflect whether or not a file
    /// was created -- only whether or not a given file’s timestamps could be successfully
    /// updated via descriptor. The `path::touched()` method wraps `detail::touch(…)`, only
    /// returning “true” iff, in the course of its call, a file is created at the path
    /// in question -- a path at which, prior to the issuing of said `detail::touch(…)` call,
    /// there wasn’t any kind of anything, filewise.
    bool path::touched() const {
        std::string thispath = str();
        bool preexisting = (::access(thispath.c_str(), F_OK) != -1);
        bool updated = detail::touch(thispath.c_str());
        return (!preexisting) && updated && (::access(thispath.c_str(), F_OK) != -1);
    }
    
    path path::resolve() const {
        /// ensure we are pointing to a symlink, or bail early:
        detail::stat_t sb;
        std::string thispath = str();
        if (::lstat(thispath.c_str(), &sb)) { return path(); }
        if (!S_ISLNK(sb.st_mode))           { return path(*this); }
        /// read the symlink’s value and return a new path:
        char linkbuf[PATH_MAX] = { 0 };
        ssize_t result = ::readlink(thispath.c_str(), linkbuf, sizeof(linkbuf));
        if (result > 0 && result < PATH_MAX) {
            return path(std::string(linkbuf, linkbuf + result));
        }
        imread_raise(FileSystemError,
            "In reference to path value:", thispath,
            "FATAL internal error in path::resolve() call to ::readlink():",
            std::strerror(errno));
    }
    
    path::size_type path::total_size() const {
        detail::nowait_t nowait;
        if (is_file_or_link()) { return filesize(); }
        if (is_directory()) {
            path::size_type out = 0;
            walk([&out](path const& p,
                        detail::stringvec_t& directories,
                        detail::stringvec_t& files) {
                std::for_each(files.begin(),
                              files.end(),
                   [&p, &out](std::string const& f) { out += (p/f).filesize(); });
            });
            return out;
        }
        return 0;
    }
    
    bool path::remove() const {
        {
            detail::nowait_t nowait;
            if (is_file_or_link()) {
                return bool(::unlink(make_absolute().c_str()) != -1);
            }
            if (is_directory()) {
                return bool(::rmdir(make_absolute().c_str()) != -1);
            }
        }
        return false;
    }
    
    bool path::rm_rf() const {
        {
            detail::nowait_t nowait;
            if (is_file_or_link()) {
                return bool(::unlink(c_str()) != -1);
            }
            if (is_directory()) {
                bool out = true;
                detail::pathvec_t dirs;
                
                /// perform walk with visitor --
                /// recursively removing files while saving directories
                /// as full paths in the `dirs` vector
                walk([&out, &dirs](path const& p,
                                   detail::stringvec_t& directories,
                                   detail::stringvec_t& files) {
                    std::for_each(directories.begin(),
                                  directories.end(),
                      [&p, &dirs](std::string const& d) {
                          dirs.emplace_back(p/d);
                    });
                    std::for_each(files.begin(),
                                  files.end(),
                       [&p, &out](std::string const& f) {
                          out &= bool(::unlink((p/f).c_str()) != -1);
                    });
                });
                
                /// remove emptied directories per saved list
                if (!dirs.empty()) {
                    std::reverse(dirs.begin(), dirs.end());         /// reverse directorylist --
                    std::for_each(dirs.begin(), dirs.end(),         /// -- removing uppermost directories top-down:
                           [&out](path const& p) {
                               out &= bool(::rmdir(p.c_str()) != -1);
                    });
                }
                
                /// return as per logical sum of `remove()` call successes
                out &= bool(::rmdir(c_str()) != -1);
                return out;
            }
        }
        return false;
    }
    
    bool path::makedir() const {
        bool out = false;
        {
            detail::nowait_t nowait;
            if (m_path.empty() || exists()) { return out; }
            out = bool(::mkdir(c_str(), detail::mkdir_flags) != -1);
        }
        return out;
    }
    
    bool path::makedir_p() const {
        /// DO NOT WAIT:
        detail::nowait_t nowait;
        
        /// sanity-check for empty or existant paths:
        if (m_path.empty() || exists()) { return false; }
        
        /// boolean to hold logical sum of operations:
        bool out = true;
        
        /// backtrack through path parents to find an existant base:
        path::size_type idx, max, i;
        idx = max = size();
        for (path p(*this); !p.exists(); p = p.parent()) { --idx; }
        
        /// copy the existant base segments to a new path instance:
        path result(detail::stringvec_t{}, m_absolute);
        result.m_path.reserve(idx);
        std::copy(m_path.begin(),
                  m_path.begin() + idx,
                  std::back_inserter(result.m_path));
        
        /// ensure the existant base exists:
        out &= result.exists();
        
        /// iterate through nonexistant path segments,
        /// using path::makedir() to make each nonexistant directory:
        if (out && idx < max) {
            for (i = idx; i < max; ++i) {
                result /= m_path[i];
                out &= bool(::mkdir(result.c_str(), detail::mkdir_flags) != -1);
            }
        }
        
        /// return per the logical sum of operations:
        return out;
    }
    
    bool path::makefifo() const {
        bool out = false;
        {
            detail::nowait_t nowait;
            if (m_path.empty() || exists()) { return out; }
            out = bool(::mkfifo(c_str(), detail::mkfifo_flags) != -1);
        }
        return out;
    }
    
    std::string path::basename() const {
        return m_path.empty() ? "" : m_path.back();
    }
    
    /// “normalizing” the path resolves and removes “.” and “..” path segments;
    /// the logic is a simplified adaptation of analagous code from TextMate:
    /// https://github.com/textmate/textmate/blob/353ae8839cf0b62dac08225a8240e9609bf0fb34/Frameworks/io/src/path.cc#L78-L112
    /// … q.v. the “toSkip” counter and the manner by which it is decremented
    path path::normalize() const {
        path out(m_absolute);       /// output path
        int skip = 0,               /// segment skip counter
            idx = m_path.size(),    /// size of segment vector
            trues = 0;              /// how many true values the vector holds
        
        /// initialize boolean vector for marking segemnts for copying:
        std::vector<bool> will_copy(idx, false);
        
        /// backtrack through path segments,
        /// only marking those for copying if they don’t resolve:
        for (; idx != -1; --idx) {
            std::string segment;
            try {
                segment = m_path.at(idx);
            } catch (std::out_of_range&) {
                continue;
            }
            
            if (!segment.size()) {
                /// noop
            } else if (segment == ".") {
                /// noop
            } else if (segment == "..") {
                ++skip;
            } else if (skip) {
                --skip;
            } else {
                will_copy[idx] = true;
                ++trues;
            }
        }
        
        /// reset the index:
        idx = 0;
        
        /// copy the marked segments to the output path for return:
        out.m_path.reserve(trues);
        std::copy_if(m_path.begin(),
                     m_path.end(),
                     std::back_inserter(out.m_path),
                 [&](std::string const& segment) { return will_copy[idx++]; });
        
        /// return the output path:
        return out;
    }
    
    bool path::rename(path const& newpath) {
        if (!exists()) { return false; }
        if (newpath.exists()) {
            path newabspath = newpath.make_absolute();
            if (!newabspath.is_directory()) { return false; }
            if (newabspath == this->parent().make_absolute()) { return false; }
            path newnewpath = newabspath.join(path(make_absolute().basename()));
            return newnewpath.exists() ? false : this->rename(newnewpath);
        }
        bool status = ::rename(str().c_str(), newpath.c_str()) == 0;
        if (status) {
            set(newpath.make_absolute().str());
        } else {
            imread_raise(FileSystemError,
                "In reference to path value:", str(), this->basename(),
                "internal error raised during path::rename() call to ::rename():",
                std::strerror(errno));
        }
        return status;
    }
    bool path::rename(char const* newpath)        { return rename(path(newpath)); }
    bool path::rename(std::string const& newpath) { return rename(path(newpath)); }
    
    path path::duplicate(path const& newpath) const {
        if (!exists()) { return path(); }
        if (newpath.exists()) {
            path newabspath = newpath.make_absolute();
            if (!newabspath.is_directory()) { return path(); }
            if (newabspath == this->parent().make_absolute()) { return path(); }
            path newnewpath = newabspath.join(path(make_absolute().basename()));
            return newnewpath.exists() ? path() : this->duplicate(newnewpath);
        }
        bool status = detail::copyfile(str().c_str(), newpath.c_str()) != -1;
        return status ? newpath.make_absolute() : path();
    }
    path path::duplicate(char const* newpath) const        { return duplicate(path(newpath)); }
    path path::duplicate(std::string const& newpath) const { return duplicate(path(newpath)); }
    
    bool path::symboliclink(path const& to) const {
        if (to.exists()) { return false; }
        if (!exists())   { return false; }
        return ::symlink(c_str(), to.c_str()) == 0;
    }
    bool path::symboliclink(char const* to) const        { return symboliclink(path(to)); }
    bool path::symboliclink(std::string const& to) const { return symboliclink(path(to)); }
    
    bool path::hardlink(path const& to) const {
        if (to.exists()) { return false; }
        if (!exists())   { return false; }
        return ::link(c_str(), to.c_str()) == 0;
    }
    bool path::hardlink(char const* to) const        { return hardlink(path(to)); }
    bool path::hardlink(std::string const& to) const { return hardlink(path(to)); }
    
    std::string path::extension() const {
        if (m_path.empty()) { return ""; }
        std::string const& last = m_path.back();
        size_type pos = last.find_last_of(path::extsep);
        if (pos == std::string::npos) { return ""; }
        return last.substr(pos+1);
    }
    
    std::string path::extensions() const {
        if (m_path.empty()) { return ""; }
        std::string const& last = m_path.back();
        size_type pos = last.find_first_of(path::extsep);
        if (pos == std::string::npos) { return ""; }
        return last.substr(pos+1);
    }
    
    path path::strip_extension() const {
        if (m_path.empty()) { return path(); }
        path result(m_path, m_absolute);
        std::string const& ext(extension());
        std::string& back(result.m_path.back());
        back = back.substr(0, back.size() - (ext.size() + 1));
        return result;
    }
    
    path path::strip_extensions() const {
        if (m_path.empty()) { return path(); }
        path result(m_path, m_absolute);
        std::string const& ext(extensions());
        std::string& back(result.m_path.back());
        back = back.substr(0, back.size() - (ext.size() + 1));
        return result;
    }
    
    detail::stringvec_t path::split_extensions() const {
        detail::stringvec_t out;
        pystring::split(extensions(), out, detail::extsepstring);
        return out;
    }
    
    path path::parent() const {
        path result;
        if (m_path.empty()) {
            if (m_absolute) {
                imread_raise(FileSystemError,
                    "path::parent() makes no sense for empty absolute paths");
            } else {
                result = path::cwd().parent();
            }
        } else {
            result.m_absolute = m_absolute;
            result.m_path.reserve(m_path.size() - 1);
            std::copy(m_path.begin(),
                      m_path.end() - 1,
                      std::back_inserter(result.m_path));
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
        result.m_path.reserve(m_path.size() + other.m_path.size());
        std::copy(other.m_path.begin(),
                  other.m_path.end(),
                  std::back_inserter(result.m_path));
        return result;
    }
    
    path& path::adjoin(path const& other) {
        if (other.m_absolute) {
            imread_raise(FileSystemError,
                "path::adjoin() expects a relative-path RHS");
        }
        m_path.reserve(m_path.size() + other.m_path.size());
        std::copy(other.m_path.begin(),
                  other.m_path.end(),
                  std::back_inserter(m_path));
        return *this;
    }
    
    path path::operator/(path const& other) const        { return join(other); }
    path path::operator/(char const* other) const        { return join(path(other)); }
    path path::operator/(std::string const& other) const { return join(path(other)); }
    
    path& path::operator/=(path const& other)        { return adjoin(other); }
    path& path::operator/=(char const* other)        { return adjoin(path(other)); }
    path& path::operator/=(std::string const& other) { return adjoin(path(other)); }
    
    path path::append(std::string const& appendix) const {
        path out(m_path.empty() ? detail::stringvec_t{ "" } : m_path, m_absolute);
        out.m_path.back().append(appendix);
        return out;
    }
    
    path& path::extend(std::string const& appendix) {
        if (appendix.empty()) {
            return *this;
        }
        if (m_path.empty()) {
            m_path = detail::stringvec_t{ "" };
        }
        m_path.back().append(appendix);
        return *this;
    }
    
    std::string&       path::operator[](size_type idx)       { return m_path[idx]; }
    std::string const& path::operator[](size_type idx) const { return m_path[idx]; }
    std::string&               path::at(size_type idx)       { return m_path.at(idx); }
    std::string const&         path::at(size_type idx) const { return m_path.at(idx); }
    
    std::string&            path::front()                    { return m_path.front(); }
    std::string const&      path::front() const              { return m_path.front(); }
    std::string&             path::back()                    { return m_path.back(); }
    std::string const&       path::back() const              { return m_path.back(); }
    
    path path::operator+(path const& other) const        { return append(other.str()); }
    path path::operator+(char const* other) const        { return append(other); }
    path path::operator+(std::string const& other) const { return append(other); }
    
    path& path::operator+=(path const& other)        { return extend(other.str()); }
    path& path::operator+=(char const* other)        { return extend(other); }
    path& path::operator+=(std::string const& other) { return extend(other); }
    
    path& path::reverse() {
        std::reverse(std::begin(m_path),
                     std::end(m_path));
        return *this;
    }
    
    path path::reversed() const {
        path out(m_absolute);
        out.m_path.reserve(m_path.size());
        std::reverse_copy(std::begin(m_path),
                          std::end(m_path),
                          std::back_inserter(out.m_path));
        return out;
    }
    
    std::string path::str() const {
        return std::accumulate(m_path.begin(),
                               m_path.end(),
                               m_absolute ? detail::sepstring
                                          : detail::nulstring,
                           [&](std::string const& lhs,
                               std::string const& rhs) {
            return lhs + rhs + ((rhs.c_str() == m_path.back().c_str()) ? detail::nulstring
                                                                       : detail::sepstring);
        });
    }
    
    char const* path::c_str() const {
        return str().c_str();
    }
    
    char const* path::data() const {
        return str().data();
    }
    
    path::size_type path::rank(std::string const& ext) const {
        /// I can't remember from whence I stole this implementation:
        std::string thispath = str();
        if (thispath.size() >= ext.size() &&
            thispath.compare(thispath.size() - ext.size(), ext.size(), ext) == 0) {
            if (thispath.size() == ext.size()) {
                return ext.size();
            }
            char ch = thispath[thispath.size() - ext.size() - 1];
            if (ch == '.' || ch == '_') {
                return ext.size() + 1;
            } else if (ch == '/') {
                return ext.size();
            }
        }
        return 0;
    }
    
    path::size_type path::rank() const {
        return path::rank(path::extension());
    }
    
    std::string path::xattr(std::string const& name) const {
        attribute::accessor_t accessor(str(), name);
        return accessor.get();
    }
    
    std::string path::xattr(std::string const& name, std::string const& value) const {
        attribute::accessor_t accessor(str(), name);
        value == filesystem::attribute::detail::nullstring ? accessor.del() : accessor.set(value);
        return accessor.get();
    }
    
    int path::xattrcount() const {
        return attribute::count(str());
    }
    
    detail::stringvec_t path::xattrs() const {
        return attribute::list(str());
    }
    
    path path::getcwd() {
        char temp[PATH_MAX];
        if (::getcwd(temp, PATH_MAX) == nullptr) {
            imread_raise(FileSystemError,
                "Internal error in getcwd():",
                std::strerror(errno));
        }
        return path(temp);
    }
    
    path path::cwd()                    { return path::getcwd(); }
    path path::gettmp()                 { return path(detail::tmpdir()); }
    path path::tmp()                    { return path(detail::tmpdir()); }
    path path::home()                   { return path(detail::userdir()); }
    path path::userdir()                { return path(detail::userdir()); }
    path path::executable()             { return path(detail::execpath()); }
    std::string path::user()            { return detail::username(); }
    std::string path::currentprogram()  { return path::basename(detail::execpath()); }
    
    detail::stringvec_t path::system() {
        return tokenize(detail::syspaths(), path::pathsep);
    }
    
    path::operator std::string() const  { return str(); }
    path::operator char const*() const  { return c_str(); }
    
    bool path::operator<(path const& rhs) const noexcept {
        return status_time() < rhs.status_time();
    }
    
    bool path::operator>(path const& rhs) const noexcept {
        return status_time() > rhs.status_time();
    }
    
    void path::set(std::string const& str) {
        m_absolute = !str.empty() && str[0] == path::sep;
        m_path = tokenize(str, path::sep);
    }
    
    path& path::operator=(std::string const& str) { set(str); return *this; }
    path& path::operator=(char const* str)        { set(str); return *this; }
    path& path::operator=(path const& p) {
        if (hash() != p.hash()) {
            path(p).swap(*this);
        }
        return *this;
    }
    
    path& path::operator=(path&& p) noexcept {
        if (hash() != p.hash()) {
            m_absolute = p.m_absolute;
            m_path = std::exchange(p.m_path, detail::stringvec_t{});
        }
        return *this;
    }
    
    path& path::operator=(detail::stringvec_t const& stringvec) {
        m_path = detail::stringvec_t{};
        m_path.reserve(stringvec.size());
        std::copy(stringvec.begin(),
                  stringvec.end(),
                  std::back_inserter(m_path));
        return *this;
    }
    
    path& path::operator=(detail::stringvec_t&& stringvec) noexcept {
        m_path = std::move(stringvec);
        return *this;
    }
    
    path& path::operator=(detail::stringlist_t stringlist) {
        m_path = detail::stringvec_t(stringlist);
        return *this;
    }
    
    std::ostream& operator<<(std::ostream& os, path const& p) {
        return os << p.str();
    }
    
    path::size_type path::hash() const noexcept {
        /// calculate the hash value for the path
        return std::accumulate(m_path.begin(),
                               m_path.end(),
                               static_cast<path::size_type>(m_absolute),
                               detail::rehasher_t());
    }
    
    void path::swap(path& other) noexcept {
        using std::swap;
        swap(m_absolute, other.m_absolute);
        swap(m_path,     other.m_path);
    }
    
    detail::stringvec_t path::components() const {
        /// return path component vector
        detail::stringvec_t out;
        out.reserve(m_path.size());
        std::copy(m_path.begin(),
                  m_path.end(),
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
    
    /// define static mutex, as declared in the switchdir struct:
    std::mutex switchdir::mute;
    
    /// define static recursive mutex, global path stack, and empty path value,
    /// as declared in the workingdir struct:
    std::recursive_mutex workingdir::mute;
    detail::pathstack_t workingdir::dstack;
    const path workingdir::empty = path();
    
} /* namespace filesystem */

namespace std {
    
    template <>
    void swap(filesystem::path& p0, filesystem::path& p1) noexcept {
        p0.swap(p1);
    }
    
    using path_hasher_t = std::hash<filesystem::path>;
    using path_arg_t = path_hasher_t::argument_type;
    using path_out_t = path_hasher_t::result_type;
    
    path_out_t path_hasher_t::operator()(path_arg_t const& p) const {
        return static_cast<path_out_t>(p.hash());
    }
    
    using inode_hasher_t = std::hash<filesystem::detail::inode_t>;
    using inode_arg_t = inode_hasher_t::argument_type;
    using inode_out_t = inode_hasher_t::result_type;
    
    /// adapted from boost methodology for std::pair<…> hashing -- q.v. sub.
    /// http://www.boost.org/doc/libs/1_64_0/doc/html/hash/reference.html#idp509100048-bb
    inode_out_t inode_hasher_t::operator()(inode_arg_t const& inode) const {
        using namespace ::hash;
        inode_out_t seed = 0;
        rehash(seed, inode.first);
        rehash(seed, inode.second);
        return seed;
    }
    
} /* namespace std */
