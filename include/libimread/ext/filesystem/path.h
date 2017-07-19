/// Copyright 2014-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_PATH_H_
#define LIBIMREAD_EXT_FILESYSTEM_PATH_H_

#include <chrono>
#include <string>
#include <vector>
#include <regex>
#include <tuple>
#include <utility>
#include <sstream>
#include <functional>
#include <initializer_list>

namespace filesystem {
    
    /// forward declaration for these next few prototypes/templates
    class path;
    
    namespace detail {
        
        /// public path-related types
        using pathvec_t = std::vector<path>;
        using pathlist_t = std::initializer_list<path>;
        using stringvec_t = std::vector<std::string>;
        using stringlist_t = std::initializer_list<std::string>;
        using vector_pair_t = std::pair<stringvec_t, stringvec_t>;      /// return type for path::list(detail::list_separate_t{})
        using clock_t = std::chrono::system_clock;
        using timepoint_t = std::chrono::time_point<clock_t>;
        using time_triple_t = std::tuple<timepoint_t, timepoint_t, timepoint_t>;
        using inode_t = std::pair<uint64_t, uint64_t>;
        
        /// utility functions that wrap system calls
        std::string tmpdir() noexcept;
        std::string userdir() noexcept;
        std::string syspaths() noexcept;
        std::string execpath() noexcept;
        timepoint_t execstarttime() noexcept;
        ssize_t copyfile(char const*, char const*);
        
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
        
        static constexpr char dotpath_extension_separator   = ':';
        static constexpr char dotpath_path_separator        = '.';
        static constexpr char dotpath_pathvar_separator     = ',';
        
        /// constant for null (nonexistent) inodes and devices:
        static const inode_t null_inode_v                   = { 0, 0 };
        
    }
    
    /// The actual class for representing a path on the filesystem
    class path {
        
        public:
            /// std::basic_string<T> analogues
            using size_type = std::string::size_type;
            using character_type = std::string::value_type;
            static constexpr character_type sep = detail::posix_path_separator;
            static constexpr character_type extsep = detail::posix_extension_separator;
            static constexpr character_type pathsep = detail::posix_pathvar_separator;
            
            /// Mapping analogues -- used with path::xattr()
            using key_type = std::string;
            using mapped_type = std::string;
            using value_type = std::pair<std::add_const_t<key_type>, mapped_type>;
            using mapping_size_type = std::size_t;
            using reference = std::add_lvalue_reference_t<value_type>;
            using const_reference = std::add_const_t<
                                    std::add_lvalue_reference_t<value_type>>;
            
            path();                                             /// Empty path constructor
            explicit path(bool);                                /// Empty but possibly absolute path constructor
            
            path(path const&);                                  /// Copy constructor
            path(path&&) noexcept;                              /// MOVE CONSTRUCTA!!
            
            path(char*);                                        /// C-string constructors
            path(char const*);
            path(std::string const&);                           /// STL-string constructor
            
            explicit path(int descriptor);                      /// File descriptor constructor
            explicit path(const void* address);                 /// Memory-address (dl_info) constructor
            
            explicit path(detail::stringvec_t const& vec,
                          bool absolute = false);               /// String-vector copy-constructor
            explicit path(detail::stringvec_t&& vec,
                          bool absolute = false) noexcept;      /// String-vector move-constructor
            explicit path(detail::stringlist_t);                /// Initializer-list constructor
            
            virtual ~path();                                    /// The path class has a vtable
            
            size_type size() const;
            bool is_absolute() const;
            bool empty() const;
            detail::inode_t inode() const;
            size_type filesize() const;
            
            /// static forwarder for path::is_absolute<P>(p)
            template <typename P> inline
            static bool is_absolute(P&& p) { return path(std::forward<P>(p)).is_absolute(); }
            
            /// static forwarder for path::inode<P>(p)
            template <typename P> inline
            static detail::inode_t inode(P&& p) { return path(std::forward<P>(p)).inode(); }
            
            /// static forwarder for path::filesize<P>(p)
            template <typename P> inline
            static size_type filesize(P&& p) { return path(std::forward<P>(p)).filesize(); }
            
            /// return a new and fully-absolute path wrapper,
            /// based on the path in question
            path make_absolute() const;
            path make_real() const;
            
            /// static forwarders for path::make_absolute<P>(p) and path::make_real<P>(p)
            template <typename P> inline
            static path absolute(P&& p) { return path(std::forward<P>(p)).make_absolute(); }
            
            template <typename P> inline
            static path real(P&& p) { return path(std::forward<P>(p)).make_real(); }
            
            /// expand a leading tilde segment -- e.g.
            ///     ~/Downloads/file.jpg
            /// becomes:
            ///     /Users/YoDogg/Downloads/file.jpg
            /// ... or whatever it is on your system, I don't know
            path expand_user() const;
            
            /// static forwarder for path::expand_user<P>(p)
            template <typename P> inline
            static path expand_user(P&& p) {
                return path(std::forward<P>(p)).expand_user();
            }
            
            bool compare_debug(path const&) const;        /// legacy, full of printf-debuggery --
                                                          /// this will throw if either path is nonexistant
            bool compare_lexical(path const&) const;      /// compare ::realpath(…) values using std::strcmp(…) --
                                                          /// this fails with nonexistant paths
            bool compare_inodes(path const&) const;       /// compare on-disk paths using detail::stat_t::st_ino values --
                                                          /// this fails with nonexistant paths
            bool compare(path const&) const noexcept;     /// compare stringified paths using fast-as-fuck path::hash() --
                                                          /// this works as expected with nonexistant paths
            
            /// static forwarders for path::compare_lexical<P, Q>(p, q), path::compare_inodes<P, Q>(p, q)
            /// and path::compare<P, Q>(p, q)
            template <typename P, typename Q> inline
            static bool compare_lexical(P&& p, Q&& q) {
                path lhs(std::forward<P>(p));
                path rhs(std::forward<Q>(q));
                if (!lhs.exists() || !rhs.exists()) { return false; }
                return lhs.compare_lexical(rhs);
            }
            
            template <typename P, typename Q> inline
            static bool compare_inodes(P&& p, Q&& q) {
                path lhs(std::forward<P>(p));
                path rhs(std::forward<Q>(q));
                if (!lhs.exists() || !rhs.exists()) { return false; }
                return lhs.compare_inodes(rhs);
            }
            
            template <typename P, typename Q> inline
            static bool compare(P&& p, Q&& q) {
                return path(std::forward<P>(p)).compare(path(std::forward<Q>(q)));
            }
            
            /// equality-test operators use path::compare(…) and therefore path::hash()
            bool operator==(path const&) const;
            bool operator!=(path const&) const;
            
            /// self-explanatory interrogatives
            bool exists() const;
            bool is_readable() const;
            bool is_writable() const;
            bool is_executable() const;
            bool is_readwritable() const;
            bool is_runnable() const;
            bool is_listable() const;
            bool is_file() const;
            bool is_link() const;
            bool is_directory() const;
            bool is_block_device() const;
            bool is_character_device() const;
            bool is_pipe() const;
            bool is_file_or_link() const;
            
            /// Static forwarders for the aforementioned interrogatives
            template <typename P> inline
            static bool exists(P&& p) { return path(std::forward<P>(p)).exists(); }
            template <typename P> inline
            static bool is_readable(P&& p) { return path(std::forward<P>(p)).is_readable(); }
            template <typename P> inline
            static bool is_writable(P&& p) { return path(std::forward<P>(p)).is_writable(); }
            template <typename P> inline
            static bool is_executable(P&& p) { return path(std::forward<P>(p)).is_executable(); }
            template <typename P> inline
            static bool is_readwritable(P&& p) { return path(std::forward<P>(p)).is_readwritable(); }
            template <typename P> inline
            static bool is_runnable(P&& p) { return path(std::forward<P>(p)).is_runnable(); }
            template <typename P> inline
            static bool is_listable(P&& p) { return path(std::forward<P>(p)).is_listable(); }
            template <typename P> inline
            static bool is_file(P&& p) { return path(std::forward<P>(p)).is_file(); }
            template <typename P> inline
            static bool is_link(P&& p) { return path(std::forward<P>(p)).is_link(); }
            template <typename P> inline
            static bool is_directory(P&& p) { return path(std::forward<P>(p)).is_directory(); }
            template <typename P> inline
            static bool is_block_device(P&& p) { return path(std::forward<P>(p)).is_block_device(); }
            template <typename P> inline
            static bool is_character_device(P&& p) { return path(std::forward<P>(p)).is_character_device(); }
            template <typename P> inline
            static bool is_pipe(P&& p) { return path(std::forward<P>(p)).is_pipe(); }
            template <typename P> inline
            static bool is_file_or_link(P&& p) { return path(std::forward<P>(p)).is_file_or_link(); }
            
            /// path::max_file_name_length() and path::max_relative_path_length()
            /// return the respective values for _PC_NAME_MAX and _PC_PATH_MAX
            /// using ::pathconf() (a value of -1 is returned for non-directories)
            long max_file_name_length() const;
            long max_relative_path_length() const;
            
            /// get timestamp values from the paths’ detail::stat_t structure,
            /// as per the system clock (std::chrono::system_clock)
            detail::time_triple_t timestamps() const;
            detail::timepoint_t access_time() const;
            detail::timepoint_t modify_time() const;
            detail::timepoint_t status_time() const;
            
            /// update the access and modification timestamps for the path
            bool update_timestamps() const;
            
            /// Static forwarder for path::timestamps<P>(p) and the other timestamp-related methods
            template <typename P> inline
            static detail::time_triple_t timestamps(P&& p) {
                return path(std::forward<P>(p)).timestamps();
            }
            
            template <typename P> inline
            static detail::timepoint_t access_time(P&& p) {
                return path(std::forward<P>(p)).access_time();
            }
            
            template <typename P> inline
            static detail::timepoint_t modify_time(P&& p) {
                return path(std::forward<P>(p)).modify_time();
            }
            
            template <typename P> inline
            static detail::timepoint_t status_time(P&& p) {
                return path(std::forward<P>(p)).status_time();
            }
            
            template <typename P> inline
            static bool update_timestamps(P&& p) {
                return path(std::forward<P>(p)).update_timestamps();
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
            
            /// static forwarder for path::match<P>(p, pattern)
            template <typename P> inline
            static bool match(P&& p, std::regex&& pattern,  bool case_sensitive=false) {
                return path(std::forward<P>(p)).match(
                    std::forward<std::regex>(pattern), case_sensitive);
            }
            
            /// static forwarder for path::search<P>(p, pattern)
            template <typename P> inline
            static bool search(P&& p, std::regex&& pattern, bool case_sensitive=false) {
                return path(std::forward<P>(p)).search(
                    std::forward<std::regex>(pattern), case_sensitive);
            }
            
            /// static forwarder for path::replace<P, S>(p, pattern, s)
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
            ///
            /// ... in all cases, you can specify a trailing boolean to ensure the paths you get back are absolute.
            /// N.B. There used to be a secondary `bool case_sensitive=false` option occasionally cluttering up
            /// the arity of these functions; now we just use `ilist(…)` to accomplsh the same.
            detail::pathvec_t     list(                             bool full_paths=false) const;
            detail::vector_pair_t list(detail::list_separate_t tag, bool full_paths=false) const;
            detail::pathvec_t     list(char const* pattern,         bool full_paths=false) const;
            detail::pathvec_t     list(std::string const& pattern,  bool full_paths=false) const;
            detail::pathvec_t     list(std::regex const& pattern,   bool full_paths=false) const; /// case-sensitive
            detail::pathvec_t    ilist(std::regex const& pattern,   bool full_paths=false) const; /// case-insensitive
            
        private:
            detail::pathvec_t  list_rx(std::regex const& pattern,   bool full_paths=false,
                                                                    bool case_sensitive=false) const;
            
        public:
            /// Generic static forwarder for permutations of path::list<P, G>(p, g)
            template <typename P, typename G> inline
            static detail::pathvec_t list(P&& p, G&& g, bool full_paths=false) {
                return path(std::forward<P>(p)).list(std::forward<G>(g), full_paths);
            }
            
            /// Generic static forwarder for permutations of path::ilist<P, G>(p, g) --
            /// at the moment, this is only a valid overload for the regex permutation of path::ilist();
            /// the other versions are backed, respectively, by glob.h and wordexp.h and,
            /// as such, aren’t readily made insensitive to case.
            template <typename P, typename G> inline
            static detail::pathvec_t ilist(P&& p, G&& g, bool full_paths=false) {
                return path(std::forward<P>(p)).ilist(std::forward<G>(g), full_paths);
            }
            
            /// Walk a path, a la os.walk() / os.path.walk() from Python
            /// ... pass a function to visit each subdirectory of a path, like so:
            /// 
            ///     path p = "/yo/dogg";
            ///     p.walk([](path const& subdir,
            ///               detail::stringvec_t& directories,
            ///               detail::stringvec_t& files) {
            ///         std::for_each(directories.begin(), directories.end(),
            ///             [&subdir](std::string const& directory) {
            ///                 std::cout << "Directory: " << subdir/directory << std::endl;
            ///         });
            ///         std::for_each(files.begin(), files.end(),
            ///             [&subdir](std::string const& file) {
            ///                 std::cout << "File: " << subdir/file << std::endl;
            ///         });
            ///     });
            /// 
            void walk(detail::walk_visitor_t&& walk_visitor) const;
            
            /// static forwarder for path::walk<P, F>(p, f)
            template <typename P, typename F> inline
            static void walk(P&& p, F&& f) {
                path(std::forward<P>(p)).walk(std::forward<F>(f));
            }
            
            /// Compute the total size of the path tree, as per the sum of the size
            /// of all files contained therein -- traversing the tree internally
            /// using a path::walk(…) visitor lambda.
            size_type total_size() const;
            
            /// Static forwarder for path::total_size<P>(p)
            template <typename P> inline
            static size_type total_size(P&& p) {
                return path(std::forward<P>(p)).total_size();
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
            
            /// Annihilate the file or directory -- recursively via path::walk(…) if the latter --
            /// in an EXTREMELY PERMANENTLY DANGEROUS manner that, really, should be USED WITH
            /// ONLY THE GREATEST EXTREMETY OF CAUTIONS EXTREME (as you would the shell command
            /// after which the method in question is named)
            bool rm_rf() const;
            
            /// Static forwarder for path::rm_rf<P>(p) that should also be USED WITH EXTREME CAUTION
            template <typename P> inline
            static bool rm_rf(P&& p) {
                return path(std::forward<P>(p)).rm_rf();
            }
            
            /// attempt to create a directory at this path. same USE-WITH-CAUTION caveats
            /// apply as per `path::remove()` (q.v. note supra).
            bool makedir() const;
            bool makedir_p() const;
            
            /// Static forwarder for path::makedir<P>(p) and path::makedir_p<P>(p) --
            /// ... again, USE WITH CAUTION people
            template <typename P> inline
            static bool makedir(P&& p) {
                return path(std::forward<P>(p)).makedir();
            }
            
            template <typename P> inline
            static bool makedir_p(P&& p) {
                return path(std::forward<P>(p)).makedir_p();
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
            
            /// create a symbolic link (via ::symlink()) from this path to another
            bool symboliclink(path const&) const;
            bool symboliclink(char const*) const;
            bool symboliclink(std::string const&) const;
            
            /// Static forwarder for path::symboliclink<P, Q>(p, q) -- (from, to) -- (source, target)
            template <typename P, typename Q> inline
            static bool symboliclink(P&& p, Q&& q) {
                return path(std::forward<P>(p)).symboliclink(std::forward<Q>(q));
            }
            
            /// create a hard link (via ::link()) from this path to another
            bool hardlink(path const&) const;
            bool hardlink(char const*) const;
            bool hardlink(std::string const&) const;
            
            /// Static forwarder for path::hardlink<P, Q>(p, q) -- (from, to) -- (source, target)
            template <typename P, typename Q> inline
            static bool hardlink(P&& p, Q&& q) {
                return path(std::forward<P>(p)).hardlink(std::forward<Q>(q));
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
            
            /// Static forwarder for path::parent<P>(p) and path::dirname<P>(p)
            template <typename P> inline
            static path parent(P&& p) {
                return path(std::forward<P>(p)).parent();
            }
            
            template <typename P> inline
            static path dirname(P&& p) {
                return path(std::forward<P>(p)).dirname();
            }
            
            /// join a path with another trailing path fragment, creating a new path:
            path join(path const& other) const;
            
            /// join a path in place with another trailing path fragment and return it:
            path& adjoin(path const& other);
            
            /// operator overloads to join paths with slashes -- you can be like this:
            ///     path p = "/yo/dogg";
            ///     path q = p / "i-heard";
            ///     path r = q / "you-like";
            ///     path s = r / "to-join-paths";
            path operator/(path const& other) const;
            path operator/(char const* other) const;
            path operator/(std::string const& other) const;
            
            /// analogous join-in-place slash-equals operators:
            path& operator/=(path const&);
            path& operator/=(char const*);
            path& operator/=(std::string const&);
            
            /// Static forwarder for path::join<P, Q>(p, q) --
            /// sometimes you want to just join stuff mainually like:
            ///     path p = path::join(p, "something/else");
            /// ... for aesthetic purposes (versus the operator overloads), etc
            template <typename P, typename Q> inline
            static path join(P&& one, Q&& theother) {
                return path(std::forward<P>(one)) / std::forward<Q>(theother);
            }
            
            /// Simple string append for the trailing path segment:
            path append(std::string const& appendix) const;
            
            /// Simple string append-in-place for the trailing path segment:
            path& extend(std::string const& appendix);
            
            /// operator overloads for bog-standard string-appending -- like so:
            ///     path p = "/yo/dogg";
            ///     path q = p + "_i_heard";
            ///     path r = q + "_you_dont_necessarily_like";
            ///     path s = r + "_segment_based_append_operations";
            path operator+(path const& other) const;
            path operator+(char const* other) const;
            path operator+(std::string const& other) const;
            
            /// analogous append-in-place plus-equals operators:
            path& operator+=(path const&);
            path& operator+=(char const*);
            path& operator+=(std::string const&);
            
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
            
            /// Filesystem extended attribute (“xattr”) access
            std::string xattr(std::string const&) const;
            std::string xattr(std::string const&, std::string const&) const;
            int xattrcount() const;
            detail::stringvec_t xattrs() const;
            
            /// Path rank value per extension
            size_type rank(std::string const&) const;
            
            /// Static forwarder for path::rank<P, S>(p, s)
            template <typename P, typename S> inline
            static size_type rank(P&& p, S&& s) {
                return path(std::forward<P>(p)).rank(std::forward<S>(s));
            }
            
            
            /// Static functions to retrieve the current directory, the system temporary directory,
            /// user/home directories, and the current running executable/program name and full path.
            static path getcwd();
            static path cwd();
            static path gettmp();
            static path tmp();
            static path home();
            static path user();
            static path executable();
            static std::string currentprogram();
            
            /// Return a vector of strings -- not paths -- corresponding to the components
            /// of the $PATH environment variable on your system (this is used internally
            /// by the filesystem::resolver class to resolve binaries as the shell would):
            static detail::stringvec_t system();
            
            /// Conversion operators -- in theory you can pass your paths to functions
            /// expecting either std::strings or const char*s with these...
            operator std::string() const;
            operator char const*() const;
            
            /// less-than operator -- allows the use of filesystem::path in e.g. std::map
            bool operator<(path const&) const noexcept;
            
            /// Set and tokenize the path using a std::string (mainly used internally)
            void set(std::string const&);
            
            /// ... and here, we have the requisite assign operators
            path& operator=(std::string const&);
            path& operator=(char const*);
            path& operator=(path const&);
            path& operator=(path&&) noexcept;
            path& operator=(detail::stringvec_t const&);
            path& operator=(detail::stringvec_t&&) noexcept;
            path& operator=(detail::stringlist_t);
            
            /// Stringify the path to an ostream
            friend std::ostream& operator<<(std::ostream&, path const&);
            
            /// calculate the hash value for the path
            size_type hash() const noexcept;
            
            /// Static forwarder for the hash function
            template <typename P> inline
            static path::size_type hash(P&& p) {
                return path(std::forward<P>(p)).hash();
            }
            
            /// no-except member swap
            void swap(path&) noexcept;
            
            /// path component vector
            detail::stringvec_t components() const;
            
        protected:
            /// Tokenize a string into a string vector:
            static detail::stringvec_t tokenize(std::string const& source,
                                                character_type const delim);
            
        protected:
            bool m_absolute = false;        /// Do we lead to our destination absolutely?
            detail::stringvec_t m_path;     /// But to where, do we eventually indexically indicate?
    
    }; /* class path */
    
}; /* namespace filesystem */

namespace std {
    
    template <>
    void swap(filesystem::path&, filesystem::path&) noexcept;
    
    /// std::hash specialization for filesystem::path
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<filesystem::path> {
        
        typedef filesystem::path argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const&) const;
        
    };
    
    template <>
    struct hash<filesystem::detail::inode_t> {
        
        typedef filesystem::detail::inode_t argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const&) const;
        
    };
    
} /* namespace std */

#endif /// LIBIMREAD_EXT_FILESYSTEM_PATH_H_