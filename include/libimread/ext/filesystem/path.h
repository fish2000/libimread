// Copyright 2015 Wenzel Jakob. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#if !defined(LIBIMREAD_EXT_FILESYSTEM_PATH_H_)
#define LIBIMREAD_EXT_FILESYSTEM_PATH_H_

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <functional>

#include <cctype>
#include <cerrno>
#include <cstring>
#include <cstddef>
#include <unistd.h>
#include <dirent.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/rehash.hh>

namespace filesystem {
    
    enum class mode { READ, WRITE };
    
    /// forward declaration for these next few prototypes/templates
    class path;
    
    namespace detail {
        
        /// Deleter structures to close directory and file handles
        template <typename D>
        struct dircloser {
            constexpr dircloser() noexcept = default;
            template <typename U> dircloser(const dircloser<U>&) noexcept {};
            void operator()(D *dirhandle) { ::closedir(dirhandle); }
        };
    
        template <typename F>
        struct filecloser {
            constexpr filecloser() noexcept = default;
            template <typename U> filecloser(const filecloser<U>&) noexcept {};
            void operator()(F *filehandle) { ::fclose(filehandle); }
        };
        
    }
    
    /// RAII-ified simplifications, for opening directory and file handles
    /// ... also known by their obnoxiously-capitalized type names
    /// dating from the dark days of early C without the plus-plus--
    /// the original opaques: FILE* and DIR*,
    /// herein wrapped neatly out of sight forever in unique_ptrs.
    /// ... YOURE WELCOME.
    using directory = std::unique_ptr<typename std::decay<DIR>::type, detail::dircloser<DIR>>;
    using file = std::unique_ptr<typename std::decay<FILE>::type, detail::filecloser<FILE>>;
    
    namespace detail {
        /// We can construct the above unique_ptr type aliases directly from FILE* and DIR* --
        /// ... these are shortcuts that wrap calls to ::opendir() and ::fopen(), respectively;
        /// so you can be like:
        ///
        ///     filesystem::directory dir = detail::ddopen("the/path/to/it/");
        ///     /// dir will auto-close on scope exit
        ///     ::some_posix_func_that_wants_a_dirhandle(dir.get());
        ///     
        /// ... see? see what I am getting at with all this? NO DIR!! haha. anyway.
        filesystem::directory ddopen(const char *c);
        filesystem::directory ddopen(const std::string &s);
        filesystem::directory ddopen(const path &p);
        filesystem::file ffopen(const std::string &s, mode m = mode::READ);
        
        /// returns a C-style string containing the temporary directory,
        // using std::getenv() and some guesswork -- originally cribbed from boost
        const char *tmpdir() noexcept;
        const char *userdir() noexcept;
        
        /// return type for path::list(), when called with a detail::list_separate_t tag
        using vector_pair_t = std::pair<std::vector<std::string>, std::vector<std::string>>;
        
        /// tag for dispatching path::list() returning detail::vector_pair_t,
        /// instead of plain ol' std::vector<path>
        struct list_separate_t {};
        
        /// user-provide callback function signature for path::walk()
        using walk_visitor_t = std::function<void(const path&,              /// root path
                                             std::vector<std::string>&,     /// directories
                                             std::vector<std::string>&)>;   /// files
        
    }
    
    /// The actual class for representing a path on the filesystem
    class path {
        
        public:
            using size_type = std::size_t;
            
            enum path_type {
                windows_path = 0, posix_path = 1,
                native_path = posix_path
            };
            
            path()
                :m_type(native_path)
                ,m_absolute(false)
                {}
            
            path(const path& p)
                :m_type(native_path)
                ,m_path(p.m_path)
                ,m_absolute(p.m_absolute)
                {}
            
            path(path&& p)
                :m_type(native_path)
                ,m_path(std::move(p.m_path))
                ,m_absolute(p.m_absolute)
                {}
            
            path(char *st)              { set(st); }
            path(const char *st)        { set(st); }
            path(const std::string &st) { set(st); }
            
            explicit path(int descriptor);
            
            inline std::size_t size() const { return static_cast<std::size_t>(m_path.size()); }
            inline bool empty() const       { return m_path.empty(); }
            inline bool is_absolute() const { return m_absolute; }
            
            /// return a new and fully-absolute path wrapper,
            /// based on the path in question
            path make_absolute() const;
            
            /// static forwarder for path::absolute<P>(p)
            template <typename P> inline
            static path absolute(P&& p) { return path(std::forward<P>(p)).make_absolute(); }
            
            /// expand a leading tilde segment -- e.g.
            ///     ~/Downloads/file.jpg
            /// becomes:
            ///     /Users/YoDogg/Downloads/file.jpg
            /// ... or whatever it is on your system, I don't know
            path expand_user() const;
            
            bool compare_debug(const path &other) const;        /// legacy, full of printf-debuggery
            bool compare_lexical(const path &other) const;      /// compare using std::strcmp(),
                                                                /// fails for nonexistant paths
            bool compare(const path &other) const;              /// compare using fast-as-fuck path::hash()
            
            /// static forwarder for path::compare<P>(p)
            template <typename P, typename Q> inline
            static bool compare(P&& p, Q&& q) {
                return path(std::forward<P>(p)).compare(path(std::forward<Q>(q)));
            }
            
            /// equality-test operators use path::hash()
            bool operator==(const path &other) const { return compare(other); }
            bool operator!=(const path &other) const { return !compare(other); }
            
            /// self-explanatory interrogatives
            bool exists() const;
            bool is_file() const;
            bool is_link() const;
            bool is_directory() const;
            bool is_file_or_link() const;
            
            /// Static forwarders for aforementioned interrogatives
            template <typename P> inline
            static bool exists(P&& p) {
                return path(std::forward<P>(p)).exists();
            }
            template <typename P> inline
            static bool is_file(P&& p) {
                return path(std::forward<P>(p)).is_file();
            }
            template <typename P> inline
            static bool is_link(P&& p) {
                return path(std::forward<P>(p)).is_link();
            }
            template <typename P> inline
            static bool is_directory(P&& p) {
                return path(std::forward<P>(p)).is_directory();
            }
            template <typename P> inline
            static bool is_file_or_link(P&& p) {
                return path(std::forward<P>(p)).is_file_or_link();
            }
            
            /// convenience funcs for matching a std::regex against the path in question:
            /// get a boolean back for your (possibly case-insenitive) std::regex reference;
            /// match() and search() hand respectively straight off to std::regex_match
            /// and std::regex_search()
            bool match(const std::regex &pattern,           bool case_sensitive=false) const;
            bool search(const std::regex &pattern,          bool case_sensitive=false) const;
            
            /// static forwarder for path::match<P>(p)
            template <typename P> inline
            static bool match(P&& p, std::regex&& pattern,  bool case_sensitive=false) {
                return path(std::forward<P>(p)).match(
                    std::forward<std::regex>(pattern), case_sensitive);
            }
            
            /// static forwarder for path::search<P>(p)
            template <typename P> inline
            static bool search(P&& p, std::regex&& pattern, bool case_sensitive=false) {
                return path(std::forward<P>(p)).search(
                    std::forward<std::regex>(pattern), case_sensitive);
            }
            
            /// list the directory contents of the path in question.
            /// The lists are stored in std::vector<path> for return.
            /// You can either:
            ///     a) pass nothing, and get all the files back -- excepting '.' and '..';
            ///     b) pass a detail::list_separate_t tag, like vanilla list(), but returns a pair of path vectors;
            ///     c) pass a string (C-style or std::string) with a glob with which to filter the list, or;
            ///     d) pass a std::regex (optionally case-sensitive) for fine-grained iterator-based filtering.
            /// ... in all cases, you can specify a trailing boolean to ensure the paths you get back are absolute.
            std::vector<path>     list(                             bool full_paths=false) const;
            detail::vector_pair_t list(detail::list_separate_t tag, bool full_paths=false) const;
            std::vector<path>     list(const char *pattern,         bool full_paths=false) const;
            std::vector<path>     list(const std::string &pattern,  bool full_paths=false) const;
            std::vector<path>     list(const std::regex &pattern,
                                       bool case_sensitive=false,   bool full_paths=false) const;
            
            /// Generic static forwarder for permutations of path::list<P, G>(p, g)
            template <typename P, typename G> inline
            static std::vector<path> list(P&& p, G&& g, bool full_paths=false) {
                return path(std::forward<P>(p)).list(std::forward<G>(g), full_paths);
            }
            
            /// Walk a path, a la os.walk() / os.path.walk() from Python
            /// ... pass a function like so:
            ///     path p = "/yo/dogg";
            ///     p.walk([](const path& p,
            ///               std::vector<std::string>& directories,
            ///               std::vector<std::string>& files) {
            ///         std::for_each(directories.begin(), directories.end(), [&p](std::string& d) {
            ///             std::cout << "Directory: " << p/d << std::endl;
            ///         });
            ///         std::for_each(files.begin(), files.end(), [&p](std::string& f) {
            ///             std::cout << "File: " << p/f << std::endl;
            ///         });
            ///     });
            void walk(detail::walk_visitor_t&& walk_visitor) const;
            
            /// static forwarder for path::walk<P, F>(p, f)
            template <typename P, typename F> inline
            static void walk(P&& p, F&& f) {
                path(std::forward<P>(p)).walk(std::forward<F>(f));
            }
            
            /// attempt to delete the file or directory at this path.
            /// USE WITH CAUTION -- this is basically ::unlink() or ::rmdir(),
            /// in other words it's like 'rm -f' on some object with which you might not
            /// be totally familiar with. Again I say: USE WITH CAUTION.
            bool remove() const;
            
            /// Static forwarder for path::remove<P>(p) that should also be USED WITH CAUTION
            template <typename P> inline
            static bool remove(P&& p) {
                return path(std::forward<P>(p)).remove();
            }
            
            /// Attempt to return the string extention (WITHOUT THE LEADING ".")
            /// ... I never know when to include the fucking leading "." and so
            /// at any given time, half my functions support it and half don't. BLEAH.
            std::string extension() const {
                if (empty()) { return ""; }
                const std::string &last = m_path[m_path.size()-1];
                std::string::size_type pos = last.find_last_of(".");
                if (pos == std::string::npos) { return ""; }
                return last.substr(pos+1);
            }
            
            /// Static forwarder for path::extension<P>(p)
            template <typename P> inline
            static std::string extension(P&& p) {
                return path(std::forward<P>(p)).extension();
            }
            
            /// Get back the parent path (also known as the 'dirname' if you are
            /// a fan of the python os.path module, which meh I could take or leave)
            path parent_path() const {
                path result;
                result.m_absolute = m_absolute;
                
                if (m_path.empty()) {
                    if (!m_absolute) { result.m_path.push_back(".."); }
                } else {
                    std::string::size_type idx = 0,
                                           until = m_path.size() - 1;
                    for (; idx < until; ++idx) {
                        result.m_path.push_back(m_path[idx]);
                    }
                }
                return result;
            }
            
            /// join a path with a new trailing path fragment
            path join(const path &other) const {
                if (other.m_absolute) {
                    imread_raise(FileSystemError,
                        "path::join() expects a relative-path RHS");
                }
                
                path result(*this);
                std::string::size_type idx = 0,
                                       max = other.m_path.size();
                
                for (; idx < max; ++idx) {
                    result.m_path.push_back(other.m_path[idx]);
                }
                return result;
            }
            
            /// operator overloads for path joining -- you can be like this:
            ///     path p = "/yo/dogg";
            ///     path q = p / "iheard";
            ///     path r = q / "youlike";
            ///     path s = r / "appending";
            path operator/(const path& other) const { return join(other); }
            path operator/(const char* other) const { return join(path(other)); }
            path operator/(const std::string& other) const { return join(path(other)); }
            
            /// Static forwarder for path::join<P, Q>(p, q) --
            /// sometimes you want to just join stuff mainually like:
            ///     path p = path::join(p, "something/else");
            /// ... for aesthetic purposes (versus the operator overloads), etc
            template <typename P, typename Q> inline
            static path join(P&& one, Q&& theother) {
                return path(std::forward<P>(one)) / path(std::forward<Q>(theother));
            }
            
            /// Simple string-append for the trailing path segment
            path append(const std::string& appendix) const {
                path out = path(*this);
                std::string::size_type N = out.m_path.size() - 1;
                out.m_path[N].append(appendix);
                return out;
            }
            
            /// operator overloads for string-appending -- like so:
            ///     path p = "/yo/dogg";
            ///     path q = p + "_i_heard";
            ///     path r = q + "_you_dont_necessarily_like";
            ///     path s = r + "_segment_based_append_operations";
            path operator+(const path& other) const { return append(other.str()); }
            path operator+(const char* other) const { return append(std::string(other)); }
            path operator+(const std::string& other) const { return append(other); }
            
            /// Static forwarder for path::append<P, Q>(p, q) --
            /// you *get* this by now, rite? It's just like some shorthand for
            ///     path p = path::append(p, ".newFileExt"); /// OR WHATEVER DUDE SRSLY UGH
            template <typename P, typename Q> inline
            static path append(P&& one, Q&& theother) {
                return path(std::forward<P>(one)) + std::forward<Q>(theother);
            }
            
            /// Stringify the path (pared down for UNIX-only specifics)
            std::string str() const {
                std::string out = "";
                const char sep = '/';
                if (m_absolute) { out += sep; }
                std::string::size_type idx = 0,
                                       siz = m_path.size();
                for (; idx < siz; ++idx) {
                    out += m_path[idx];
                    if (idx + 1 < siz) { out += sep; }
                }
                return out;
            }
            
            /// Convenience function to get a C-style string, a la std::string's API
            inline const char *c_str() const { return str().c_str(); }
            
            /// Static functions for getting both the current, system temp, and user/home directories
            static path getcwd();
            static path cwd()               { return path::getcwd(); }
            static path gettmp()            { return path(detail::tmpdir()); }
            static path tmp()               { return path(detail::tmpdir()); }
            static path home()              { return path(detail::userdir()); }
            static path user()              { return path(detail::userdir()); }
            
            /// Conversion operators -- in theory you can pass your paths to functions
            /// expecting either std::strings or const char*s with these...
            operator std::string()          { return str(); }
            operator const char*()          { return c_str(); }
            
            /// Set and tokenize the path using a std::string (mainly used internally)
            void set(const std::string &str) {
                m_type = native_path;
                m_path = tokenize(str, "/");
                m_absolute = !str.empty() && str[0] == '/';
            }
            
            /// ... and here, we have the requisite assign operators
            path &operator=(const std::string &str) { set(str); return *this; }
            path &operator=(const char *str)        { set(str); return *this; }
            path &operator=(const path& p) {
                if (!compare(p, *this)) {
                    path(p).swap(*this);
                }
                return *this;
            }
            path &operator=(path&& p) {
                m_type = native_path;
                m_absolute = p.m_absolute;
                if (!compare(p, *this)) {
                    m_path = std::move(p.m_path);
                }
                return *this;
            }
            
            /// Stringify the path to an ostream
            friend std::ostream &operator<<(std::ostream &os, const path &p) {
                return os << p.str();
            }
            
            /// calculate the hash value for the path
            std::size_t hash() const {
                std::size_t H = static_cast<std::size_t>(is_absolute());
                for (auto idx = m_path.begin(); idx != m_path.end(); ++idx) {
                    ::detail::rehash(H, *idx);
                }
                return H;
            }
            
            /// Static forwarder for the hash function
            template <typename P> inline
            static std::size_t hash(P&& p) {
                return path(std::forward<P>(p)).hash();
            }
            
            /// no-except member swap
            void swap(path& other) noexcept;
            
        protected:
            static std::vector<std::string> tokenize(const std::string &source, const std::string &delim) {
                std::vector<std::string> tokens;
                std::string::size_type lastPos = 0,
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
            
        protected:
            path_type m_type;
            std::vector<std::string> m_path;
            bool m_absolute;
    
    }; /* class path */
    
    /// change directory temporarily with RAII
    struct switchdir {
        
        explicit switchdir(path nd)
            :olddir(path::cwd().str())
            ,newdir(nd.str())
            {
                mute.lock();
                ::chdir(newdir.c_str());
            }
        
        path from() const { return path(olddir); }
        
        ~switchdir() {
            ::chdir(olddir.c_str());
            mute.unlock();
        }
        
        private:
            switchdir(void);
            switchdir(const switchdir&);
            switchdir(switchdir&&);
            switchdir &operator=(const switchdir&);
            switchdir &operator=(switchdir&&);
            static std::mutex mute; /// declaration but not definition
            mutable std::string olddir;
            mutable std::string newdir;
    };
    
}; /* namespace filesystem */

namespace std {
    
    template <>
    void swap(filesystem::path& p0, filesystem::path& p1);
    
    /// std::hash specialization for filesystem::path
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<filesystem::path> {
        
        typedef filesystem::path argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& p) const {
            return static_cast<result_type>(p.hash());
        }
        
    };
    
}; /* namespace std */

#endif /// LIBIMREAD_EXT_FILESYSTEM_PATH_H_
