// Copyright 2015 Wenzel Jakob. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef LIBIMREAD_EXT_FILESYSTEM_PATH_H_
#define LIBIMREAD_EXT_FILESYSTEM_PATH_H_

#include <string>
#include <vector>
#include <regex>
#include <utility>
#include <sstream>
#include <functional>
#include <initializer_list>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>

namespace filesystem {
    
    /// forward declaration for these next few prototypes/templates
    class path;
    
    namespace detail {
        /// returns a C-style string containing the temporary directory,
        // using std::getenv() and some guesswork -- originally cribbed from boost
        char const* tmpdir() noexcept;
        char const* userdir() noexcept;
        char const* syspaths() noexcept;
        std::string execpath() noexcept;
        ssize_t copyfile(char const*, char const*);
        
        /// return type for path::list(), when called with a detail::list_separate_t tag
        using pathvec_t = std::vector<path>;
        using pathlist_t = std::initializer_list<path>;
        using stringvec_t = std::vector<std::string>;
        using stringlist_t = std::initializer_list<std::string>;
        using vector_pair_t = std::pair<stringvec_t, stringvec_t>;
        using inode_t = uint64_t;
        
        /// tag for dispatching path::list() returning detail::vector_pair_t,
        /// instead of plain ol' pathvec_t
        struct list_separate_t {};
        
        /// user-provided callback function signature for path::walk()
        using walk_visitor_t = std::function<void(path const&,  /// root path
                                             stringvec_t&,      /// directories
                                             stringvec_t&)>;    /// files
        
        /// constants for path separators
        static constexpr char windows_extension_separator = '.';
        static constexpr char windows_path_separator      = '\\';
        static constexpr char windows_pathvar_separator   = ';';
        static constexpr char posix_extension_separator   = '.';
        static constexpr char posix_path_separator        = '/';
        static constexpr char posix_pathvar_separator     = ':';
        
        /// constant for null (nonexistent) inode
        static constexpr inode_t null_inode_v             = 0;
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
            
            path();
            path(path const& p);
            path(path&& p) noexcept;
            
            path(char* st);
            path(char const* st);
            path(std::string const& st);
            
            explicit path(int descriptor);
            explicit path(const void* address);
            explicit path(detail::stringvec_t const& vec, bool absolute = false);
            explicit path(detail::stringlist_t list);
            
            std::size_t size() const;
            bool is_absolute() const;
            bool empty() const;
            detail::inode_t inode() const;
            std::size_t filesize() const;
            
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
            
            /// static forwarder for path::expand_user<P>(p)
            template <typename P> inline
            static path expand_user(P&& p) { return path(std::forward<P>(p)).expand_user(); }
            
            bool compare_debug(path const& other) const;        /// legacy, full of printf-debuggery
            bool compare_lexical(path const& other) const;      /// compare using std::strcmp(),
                                                                /// fails for nonexistant paths
            bool compare(path const& other) const noexcept;     /// compare using fast-as-fuck path::hash()
            
            /// static forwarder for path::compare<P>(p)
            template <typename P, typename Q> inline
            static bool compare(P&& p, Q&& q) {
                return path(std::forward<P>(p)).compare(path(std::forward<Q>(q)));
            }
            
            /// equality-test operators use path::hash()
            bool operator==(path const& other) const { return compare(other); }
            bool operator!=(path const& other) const { return !compare(other); }
            
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
            bool match(std::regex const& pattern,           bool case_sensitive=false) const;
            bool search(std::regex const& pattern,          bool case_sensitive=false) const;
            path replace(std::regex const& pattern,         char const* replacement,
                                                            bool case_sensitive=false) const;
            path replace(std::regex const& pattern,         std::string const& replacement,
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
            detail::pathvec_t     list(char const* pattern,         bool full_paths=false) const;
            detail::pathvec_t     list(std::string const& pattern,  bool full_paths=false) const;
            detail::pathvec_t     list(std::regex const& pattern,
                                       bool case_sensitive=false,   bool full_paths=false) const;
            
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
            void walk(detail::walk_visitor_t&& walk_visitor) const;
            
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
            
            /// attempt to create a directory at this path. same USE-WITH-CAUTION caveats
            /// apply as per `path::remove()` (q.v. note supra).
            bool makedir() const;
            
            /// Static forwarder for path::makedir<P>(p) -- again, USE WITH CAUTION people
            template <typename P> inline
            static bool makedir(P&& p) {
                return path(std::forward<P>(p)).makedir();
            }
            
            /// get the basename -- i.e. for path /yo/dogg/iheardyoulike/basenames.jpg
            /// ... the basename returned is "basenames.jpg"
            std::string basename() const;
            
            /// Static forwarder for path::basename<P>(p)
            template <typename P> inline
            static std::string basename(P&& p) {
                return path(std::forward<P>(p)).basename();
            }
            
            /// rename a file (using ::rename()),
            /// specifying the new name with a new path instance
            bool rename(path const& newpath);
            bool rename(char const* newpath);
            bool rename(std::string const& newpath);
            
            /// Static forwarder for path::rename<P, Q>(p, q)
            template <typename P, typename Q> inline
            static bool rename(P&& p, Q&& q) {
                return path(std::forward<P>(p)).rename(std::forward<Q>(q));
            }
            
            /// duplicate a file (using ::fcopyfile() or ::sendfile()),
            /// specifying the new name with a new path instance
            path duplicate(path const& newpath) const;
            path duplicate(char const* newpath) const;
            path duplicate(std::string const& newpath) const;
            
            /// Static forwarder for path::rename<P, Q>(p, q)
            template <typename P, typename Q> inline
            static path duplicate(P&& p, Q&& q) {
                return path(std::forward<P>(p)).duplicate(std::forward<Q>(q));
            }
            
            /// Attempt to return the string extention (WITHOUT THE LEADING ".")
            /// ... I never know when to include the fucking leading "." and so
            /// at any given time, half my functions support it and half don't. BLEAH.
            std::string extension() const;
            std::string extensions() const;
            
            /// Static forwarder for path::extension<P>(p) and path::extensions<P>(p)
            template <typename P> inline
            static std::string extension(P&& p) {
                return path(std::forward<P>(p)).extension();
            }
            
            template <typename P> inline
            static std::string extensions(P&& p) {
                return path(std::forward<P>(p)).extensions();
            }
            
            /// Return a new path with the extension stripped off, e.g.
            ///  path("/yo/dogg.tar.gz").strip_extension() == path("/yo/dogg.tar");
            /// path("/yo/dogg.tar.gz").strip_extensions() == path("/yo/dogg");
            path strip_extension() const;
            path strip_extensions() const;
            
            /// Static forwarder for path::strip_extension<P>(p) and path::strip_extensions<P>(p)
            template <typename P> inline
            static path strip_extension(P&& p) {
                return path(std::forward<P>(p)).strip_extension();
            }
            
            template <typename P> inline
            static path strip_extensions(P&& p) {
                return path(std::forward<P>(p)).strip_extensions();
            }
            
            /// Convenience method to get the result of path::extensions(),
            /// split on path::extsep, as a vector of strings.
            detail::stringvec_t split_extensions() const;
            
            /// Static forwarder for path::split_extensions<P>(p)
            template <typename P> inline
            static detail::stringvec_t split_extensions(P&& p) {
                return path(std::forward<P>(p)).split_extensions();
            }
            
            /// Get back the parent path (also known as the 'dirname' if you are
            /// a fan of the Python os.path module, which meh I could take or leave)
            path parent() const;
            path dirname() const;
            
            /// join a path with a new trailing path fragment
            path join(path const& other) const;
            
            /// operator overloads to join paths with slashes -- you can be like this:
            ///     path p = "/yo/dogg";
            ///     path q = p / "i-heard";
            ///     path r = q / "you-like";
            ///     path s = r / "to-join-paths";
            path operator/(path const& other) const;
            path operator/(char const* other) const;
            path operator/(std::string const& other) const;
            
            /// Static forwarder for path::join<P, Q>(p, q) --
            /// sometimes you want to just join stuff mainually like:
            ///     path p = path::join(p, "something/else");
            /// ... for aesthetic purposes (versus the operator overloads), etc
            template <typename P, typename Q> inline
            static path join(P&& one, Q&& theother) {
                return path(std::forward<P>(one)) / path(std::forward<Q>(theother));
            }
            
            /// Simple string-append for the trailing path segment
            path append(std::string const& appendix) const;
            
            /// operator overloads for bog-standard string-appending -- like so:
            ///     path p = "/yo/dogg";
            ///     path q = p + "_i_heard";
            ///     path r = q + "_you_dont_necessarily_like";
            ///     path s = r + "_segment_based_append_operations";
            path operator+(path const& other) const;
            path operator+(char const* other) const;
            path operator+(std::string const& other) const;
            
            /// Static forwarder for path::append<P, Q>(p, q) --
            /// you *get* this by now, rite? It's just like some shorthand for
            ///     path p = path::append(p, ".newFileExt"); /// OR WHATEVER DUDE SRSLY UGH
            template <typename P, typename Q> inline
            static path append(P&& one, Q&& theother) {
                return path(std::forward<P>(one)) + std::forward<Q>(theother);
            }
            
            /// Stringify the path (pared down for UNIX-only specifics)
            std::string str() const;
            
            /// Convenience function to get a C-style string, a la std::string's API
            char const* c_str() const;
            
            /// Static functions for getting both the current, system temp, and user/home directories
            static path getcwd();
            static path cwd();
            static path gettmp();
            static path tmp();
            static path home();
            static path user();
            static path executable();
            
            static detail::stringvec_t system();
            
            /// Conversion operators -- in theory you can pass your paths to functions
            /// expecting either std::strings or const char*s with these...
            operator std::string()           { return str(); }
            operator char const*()           { return c_str(); }
            
            /// less-than operator -- allows the use of filesystem::path in e.g. std::map
            bool operator<(path const& rhs) const noexcept;
            
            /// Set and tokenize the path using a std::string (mainly used internally)
            void set(std::string const& str);
            
            /// ... and here, we have the requisite assign operators
            path& operator=(std::string const& str);
            path& operator=(char const* str);
            path& operator=(path const& p);
            path& operator=(path&& p) noexcept;
            
            /// Stringify the path to an ostream
            friend std::ostream& operator<<(std::ostream& os, path const& p);
            
            /// calculate the hash value for the path
            std::size_t hash() const noexcept;
            
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
            static detail::stringvec_t tokenize(std::string const& source,
                                                character_type const delim);
            
            path_type m_type;
            detail::stringvec_t m_path;
            bool m_absolute;
    
    }; /* class path */
    
}; /* namespace filesystem */

namespace std {
    
    template <>
    void swap(filesystem::path& p0, filesystem::path& p1) noexcept;
    
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
