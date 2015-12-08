// Copyright 2015 Wenzel Jakob. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIBIMREAD_EXT_FILESYSTEM_PATH_H_
#define LIBIMREAD_EXT_FILESYSTEM_PATH_H_

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <regex>
#include <stack>
#include <sstream>
#include <utility>
#include <exception>
#include <functional>
#include <initializer_list>

#include <unistd.h>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>

namespace filesystem {
    
    enum class mode { READ, WRITE };
    
    /// forward declaration for these next few prototypes/templates
    class path;
    
    namespace detail {
        /// returns a C-style string containing the temporary directory,
        // using std::getenv() and some guesswork -- originally cribbed from boost
        const char* tmpdir() noexcept;
        const char* userdir() noexcept;
        const char* syspaths() noexcept;
        std::string execpath() noexcept;
        
        /// return type for path::list(), when called with a detail::list_separate_t tag
        using pathvec_t = std::vector<path>;
        using pathlist_t = std::initializer_list<path>;
        using stringvec_t = std::vector<std::string>;
        using stringlist_t = std::initializer_list<std::string>;
        using vector_pair_t = std::pair<stringvec_t, stringvec_t>;
        
        /// tag for dispatching path::list() returning detail::vector_pair_t,
        /// instead of plain ol' pathvec_t
        struct list_separate_t {};
        
        /*
        struct halt_walk : std::exception {
            explicit halt_walk(std::size_t L = 0)
                :std::exception(),
                ,initial_level(L), level(L)
                {}
            halt_walk& operator++() noexcept {
                level++;
                return *this;
            }
            halt_walk& operator--() noexcept {
                if (level > 0) { level--; }
                return *this;
            }
            halt_walk operator++(int) noexcept {
                ++level;
                return *this;
            }
            halt_walk operator--(int) noexcept {
                if (level > 0) { --level; }
                return *this;
            }
            std::size_t& level() const noexcept { return &level; }
            std::size_t const& initial_level() const noexcept { return &initial_level; }
            char const* what() const noexcept {
                std::ostringstream out;
                out << "Walk halted, level: "         << level << ", " << std::endl
                    << "             initial level: " << initial_level << std::endl;
                return out.str().c_str();
            }
            private:
                std::size_t level;
                std::size_t initial_level;
        };
        */
        
        struct halt_walk : std::runtime_error {
            explicit halt_walk(std::size_t level)
                :initial_level(level)
                ,liljon("Walk halted, level: ")
                { liljon += std::to_string(level); }
            char const* what() const noexcept { return liljon.c_str(); }
            private:
                std::size_t initial_level;
                std::string liljon;
        };
        
        /// user-provided callback function signature for path::walk()
        using walk_visitor_t = std::function<void(const path&,  /// root path
                                             stringvec_t&,      /// directories
                                             stringvec_t&)>;    /// files
        
        /// constants for path separators
        static constexpr char windows_extension_separator = '.';
        static constexpr char windows_path_separator      = '\\';
        static constexpr char windows_pathvar_separator   = ';';
        static constexpr char posix_extension_separator   = '.';
        static constexpr char posix_path_separator        = '/';
        static constexpr char posix_pathvar_separator     = ':';
    }
    
    /// The actual class for representing a path on the filesystem
    class path {
        
        public:
            using size_type = std::string::size_type;
            using character_type = std::string::value_type;
            static constexpr character_type sep = detail::posix_path_separator;
            static constexpr character_type extsep = detail::posix_extension_separator;
            
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
            
            path(char* st)              { set(st); }
            path(const char* st)        { set(st); }
            path(const std::string& st) { set(st); }
            
            explicit path(int descriptor);
            explicit path(detail::stringlist_t list);
            
            inline std::size_t size() const { return static_cast<std::size_t>(m_path.size()); }
            inline bool is_absolute() const { return m_absolute; }
            inline bool empty() const       { return m_path.empty(); }
            
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
            
            bool compare_debug(const path& other) const;        /// legacy, full of printf-debuggery
            bool compare_lexical(const path& other) const;      /// compare using std::strcmp(),
                                                                /// fails for nonexistant paths
            bool compare(const path& other) const;              /// compare using fast-as-fuck path::hash()
            
            /// static forwarder for path::compare<P>(p)
            template <typename P, typename Q> inline
            static bool compare(P&& p, Q&& q) {
                return path(std::forward<P>(p)).compare(path(std::forward<Q>(q)));
            }
            
            /// equality-test operators use path::hash()
            bool operator==(const path& other) const { return compare(other); }
            bool operator!=(const path& other) const { return !compare(other); }
            
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
            
            /// Convenience funcs for running a std::regex against the path in question:
            /// match(), search() and replace() hand respectively straight off to std::regex_match,
            /// std::regex_search(), and std::regex_replace, using the stringified path.
            /// As one would expect, calls to the former two return booleans whereas
            /// both replace() overloads yield new path objects.
            bool match(const std::regex& pattern,           bool case_sensitive=false) const;
            bool search(const std::regex& pattern,          bool case_sensitive=false) const;
            path replace(const std::regex& pattern,         const char* replacement,
                                                            bool case_sensitive=false) const;
            path replace(const std::regex& pattern,         const std::string& replacement,
                                                            bool case_sensitive=false) const;
            
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
            
            /// static forwarder for path::replace<P>(p)
            template <typename P, typename S> inline
            static path replace(P&& p, std::regex&& pattern, S&& s, bool case_sensitive=false) {
                return path(std::forward<P>(p)).replace(
                    std::forward<std::regex>(pattern), std::forward<S>(s), case_sensitive);
            }
            
            /// list the directory contents of the path in question.
            /// The lists are stored in std::vector<path> for return.
            /// You can either:
            ///     a) pass nothing, and get all the files back -- excepting '.' and '..';
            ///     b) pass a detail::list_separate_t tag, like vanilla list(), but returns a pair of path vectors;
            ///     c) pass a string (C-style or std::string) with a glob with which to filter the list, or;
            ///     d) pass a std::regex (optionally case-sensitive) for fine-grained iterator-based filtering.
            /// ... in all cases, you can specify a trailing boolean to ensure the paths you get back are absolute.
            detail::pathvec_t     list(                             bool full_paths=false) const;
            detail::vector_pair_t list(detail::list_separate_t tag, bool full_paths=false) const;
            detail::pathvec_t     list(const char* pattern,         bool full_paths=false) const;
            detail::pathvec_t     list(const std::string& pattern,  bool full_paths=false) const;
            detail::pathvec_t     list(const std::regex& pattern,
                                       bool case_sensitive=false,   bool full_paths=false) const;
            
            /// Generic static forwarder for path::list<P>(p)
            // template <typename P> inline
            // static detail::pathvec_t list(P&& p, bool full_paths=false) {
            //     return path(std::forward<P>(p)).list(full_paths);
            // }
            
            /// Generic static forwarder for permutations of path::list<P, G>(p, g)
            template <typename P, typename G> inline
            static detail::pathvec_t list(P&& p, G&& g, bool full_paths=false) {
                return path(std::forward<P>(p)).list(std::forward<G>(g), full_paths);
            }
            
            /// Walk a path, a la os.walk() / os.path.walk() from Python
            /// ... pass a function like so:
            ///     path p = "/yo/dogg";
            ///     p.walk([](const path& p,
            ///               detail::stringvec_t& directories,
            ///               detail::stringvec_t& files) {
            ///         std::for_each(directories.begin(), directories.end(), [&p](std::string& d) {
            ///             std::cout << "Directory: " << p/d << std::endl;
            ///         });
            ///         std::for_each(files.begin(), files.end(), [&p](std::string& f) {
            ///             std::cout << "File: " << p/f << std::endl;
            ///         });
            ///     });
            void walk(detail::walk_visitor_t&& walk_visitor,
                      std::size_t level = 0) const;
            
            /// static forwarder for path::walk<P, F>(p, f)
            template <typename P, typename F> inline
            static void walk(P&& p, F&& f) {
                path(std::forward<P>(p)).walk(std::forward<F>(f));
            }
            
            /// attempt to delete the file or directory at this path.
            /// USE WITH CAUTION -- this is basically ::unlink() or ::rmdir(),
            /// in other words it's like 'rm -f' on some object with which you might not
            /// be totally familiar. Again I say: USE WITH CAUTION.
            bool remove() const;
            
            /// Static forwarder for path::remove<P>(p) that should also be USED WITH CAUTION
            template <typename P> inline
            static bool remove(P&& p) {
                return path(std::forward<P>(p)).remove();
            }
            
            /// get the basename -- i.e. for path /yo/dogg/iheardyoulike/basenames.jpg
            /// ... the basename returned is "basenames.jpg"
            std::string basename() const {
                if (empty()) { return ""; }
                return m_path.back();
            }
            
            /// Static forwarder for path::basename<P>(p)
            template <typename P> inline
            static std::string basename(P&& p) {
                return path(std::forward<P>(p)).basename();
            }
            
            /// Attempt to return the string extention (WITHOUT THE LEADING ".")
            /// ... I never know when to include the fucking leading "." and so
            /// at any given time, half my functions support it and half don't. BLEAH.
            std::string extension() const {
                if (empty()) { return ""; }
                const std::string& last = m_path.back();
                size_type pos = last.find_last_of(extsep);
                if (pos == std::string::npos) { return ""; }
                return last.substr(pos+1);
            }
            
            /// Static forwarder for path::extension<P>(p)
            template <typename P> inline
            static std::string extension(P&& p) {
                return path(std::forward<P>(p)).extension();
            }
            
            /// Get back the parent path (also known as the 'dirname' if you are
            /// a fan of the Python os.path module, which meh I could take or leave)
            path parent() const {
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
            
            inline path dirname() const { return parent(); }
            
            /// join a path with a new trailing path fragment
            path join(const path& other) const {
                if (other.m_absolute) {
                    imread_raise(FileSystemError,
                        "path::join() expects a relative-path RHS");
                }
                
                path result(*this);
                size_type idx = 0,
                          max = other.m_path.size();
                
                for (; idx < max; ++idx) {
                    result.m_path.push_back(other.m_path[idx]);
                }
                return result;
            }
            
            /// operator overloads to join paths with slashes -- you can be like this:
            ///     path p = "/yo/dogg";
            ///     path q = p / "iheard";
            ///     path r = q / "youlike";
            ///     path s = r / "appending";
            path operator/(const path& other) const        { return join(other); }
            path operator/(const char* other) const        { return join(path(other)); }
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
                out.m_path.back().append(appendix);
                return out;
            }
            
            /// operator overloads for bog-standard string-appending -- like so:
            ///     path p = "/yo/dogg";
            ///     path q = p + "_i_heard";
            ///     path r = q + "_you_dont_necessarily_like";
            ///     path s = r + "_segment_based_append_operations";
            path operator+(const path& other) const        { return append(other.str()); }
            path operator+(const char* other) const        { return append(other); }
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
                if (m_absolute) { out += sep; }
                size_type idx = 0,
                          siz = m_path.size();
                for (; idx < siz; ++idx) {
                    out += m_path[idx];
                    if (idx + 1 < siz) { out += sep; }
                }
                return out;
            }
            
            /// Convenience function to get a C-style string, a la std::string's API
            inline const char* c_str() const { return str().c_str(); }
            
            /// Static functions for getting both the current, system temp, and user/home directories
            static path getcwd();
            static path cwd()                { return path::getcwd(); }
            static path gettmp()             { return path(detail::tmpdir()); }
            static path tmp()                { return path(detail::tmpdir()); }
            static path home()               { return path(detail::userdir()); }
            static path user()               { return path(detail::userdir()); }
            static path executable()         { return path(detail::execpath()); }
            
            static detail::stringvec_t system() {
                return tokenize(detail::syspaths(), detail::posix_pathvar_separator);
            }
            
            /// Conversion operators -- in theory you can pass your paths to functions
            /// expecting either std::strings or const char*s with these...
            operator std::string()           { return str(); }
            operator const char*()           { return c_str(); }
            
            /// Set and tokenize the path using a std::string (mainly used internally)
            void set(const std::string& str) {
                m_type = native_path;
                m_path = tokenize(str, sep);
                m_absolute = !str.empty() && str[0] == sep;
            }
            
            /// ... and here, we have the requisite assign operators
            path &operator=(const std::string& str) { set(str); return *this; }
            path &operator=(const char* str)        { set(str); return *this; }
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
            friend std::ostream &operator<<(std::ostream& os, const path& p) {
                return os << p.str();
            }
            
            /// calculate the hash value for the path
            std::size_t hash() const;
            
            /// Static forwarder for the hash function
            template <typename P> inline
            static std::size_t hash(P&& p) {
                return path(std::forward<P>(p)).hash();
            }
            
            /// no-except member swap
            void swap(path& other) noexcept;
            
            /// path component vector
            detail::stringvec_t components() const;
            
        protected:
            static detail::stringvec_t tokenize(const std::string& source,
                                                const character_type delim) {
                detail::stringvec_t tokens;
                size_type lastPos = 0,
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
            detail::stringvec_t m_path;
            bool m_absolute;
    
    }; /* class path */
    
    
    struct switchdir {
        
        /// Change working directory temporarily with RAII while
        /// holding a process-unique lock during the switchover.
        ///
        ///     // assume we ran as "cd /usr/local/bin && ./yodogg"
        ///     using filesystem::path;
        ///     path new_directory = path("/usr/local/var/myshit");
        ///     // most list() calls yield pathvec_t, a std::vector<filesystem::path>
        ///     filesystem::detail::pathvec_t stuff;
        ///     {
        ///         filesystem::switchdir switch(new_directory); // threads block here serially
        ///         assert(path::cwd() == "/usr/local/var/myshit");
        ///         assert(switchdir.from() == "/usr/local/bin");
        ///         stuff = path::cwd().list();
        ///     }
        ///     // scope exit destroys the `switch` instance --
        ///     // restoring the process-wide previous working directory
        ///     // and unlocking the global `filesystem::switchdir` mutex
        ///     assert(path::cwd() == "/usr/local/bin");
        ///     std::cout << "We found " << stuff.size()
        ///               << " items inside  " new_directory << "..."
        ///               << std::endl;
        ///     std::cout << "We're currently working out of " << path::cwd() << ""
        ///               << std::endl;
        ///
        /// This avoids a slew of the race conditions you are risking
        /// whenever you start shooting off naked ::chdir() calls --
        /// including wierd results from APIs like the POSIX filesystem calls
        /// (e.g. ::glob() and ::readdir(), both of which path.cpp leverages).
        /// Those older C-string-based interfaces are generous with
        /// semantic vagaries, and can behave in ways that make irritating
        /// use of, or assumptions about, the process' current working directory.
        /// ... and so yeah: "Block before chdir()" is the new "Use a condom".
        
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
            static std::mutex mute; /// declaration not definition
            mutable std::string olddir;
            mutable std::string newdir;
    };
    
    struct workingdir {
        
        /// Change working directory multiple times with a RAII idiom,
        /// using a process-unique recursive lock to maintain a stack of
        /// previously occupied directories.
        /// rewinding automatically to the previous originating directory
        /// on scope exit.
        
        workingdir(path&& nd)
            { push(std::forward<path>(nd)); }
        
        path from() const { return path(top()); }
        ~workingdir() { pop(); }
        
        static void push(path&& nd) {
            if (nd == dstack.top()) { return; }
            mute.lock();
            dstack.push(path::cwd());
            ::chdir(nd.c_str());
        }
        
        static void pop() {
            if (dstack.empty()) { return; }
            mute.unlock();
            ::chdir(dstack.top().c_str());
            dstack.pop();
        }
        
        static const path& top() {
            return dstack.empty() ? empty : dstack.top();
        }
        
        private:
            workingdir(void);
            workingdir(const workingdir&);
            workingdir(workingdir&&);
            workingdir &operator=(const workingdir&);
            workingdir &operator=(workingdir&&);
            static std::recursive_mutex mute; /// declaration not definition
            static std::stack<path> dstack;   /// declaration not definition
            static const path empty;          /// declaration not definition
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
