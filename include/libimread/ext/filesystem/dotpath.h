/// Copyright 2014-2018 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_DOTPATH_H_
#define LIBIMREAD_EXT_FILESYSTEM_DOTPATH_H_

#include <string>
#include <vector>
#include <regex>
#include <utility>
#include <sstream>
#include <functional>
#include <initializer_list>

namespace filesystem {
    
    /// forward declaration for these next few prototypes/templates
    class path;
    class dotpath;
    
    namespace detail {
        
        /// public dotpath-related types
        using dotpathvec_t = std::vector<dotpath>;
        using dotpathlist_t = std::initializer_list<dotpath>;
        using stringvec_t = std::vector<std::string>;
        using stringlist_t = std::initializer_list<std::string>;
        
        static constexpr char dotpath_extension_separator   = ':';
        static constexpr char dotpath_path_separator        = '.';
        static constexpr char dotpath_pathvar_separator     = ',';
        
    }
    
    /// The actual class for representing a dotpath
    class dotpath {
        
        public:
            /// std::basic_string<T> analogues
            using size_type = std::string::size_type;
            using character_type = std::string::value_type;
            static constexpr character_type sep = detail::dotpath_path_separator;
            static constexpr character_type extsep = detail::dotpath_extension_separator;
            static constexpr character_type pathsep = detail::dotpath_pathvar_separator;
            
            dotpath();                                          /// Empty dotpath constructor
            explicit dotpath(bool);                             /// Empty but possibly absolute dotpath constructor
            
            dotpath(dotpath const&);                            /// Copy constructor
            dotpath(dotpath&&) noexcept;                        /// MOVE CONSTRUCTA!!
            
            explicit dotpath(path const&);                      /// Convert from path const& (reference)
            explicit dotpath(path&&) noexcept;                  /// Convert from path&& (temporary)
            
            dotpath(char*);                                     /// C-string constructors
            dotpath(char const*);
            dotpath(std::string const&);                        /// STL-string constructor
            
            explicit dotpath(detail::stringvec_t const& vec,
                             bool absolute = false);            /// String-vector copy-constructor
            explicit dotpath(detail::stringvec_t&& vec,
                             bool absolute = false) noexcept;   /// String-vector move-constructor
            explicit dotpath(detail::stringlist_t);             /// Initializer-list constructor
            
            virtual ~dotpath();                                 /// The dotpath class has a vtable
            
            size_type size() const;
            size_type length() const;
            bool is_absolute() const;
            bool empty() const;
            
            /// static forwarder for dotpath::is_absolute<P>(p)
            template <typename P> inline
            static bool is_absolute(P&& p) { return dotpath(std::forward<P>(p)).is_absolute(); }
            
            /// return a new and fully-absolute dotpath wrapper,
            /// based on the dotpath in question
            dotpath make_absolute() const;
            dotpath make_real() const;
            
            /// static forwarders for dotpath::make_absolute<P>(p) and dotpath::make_real<P>(p)
            template <typename P> inline
            static dotpath absolute(P&& p) { return dotpath(std::forward<P>(p)).make_absolute(); }
            
            template <typename P> inline
            static dotpath real(P&& p) { return dotpath(std::forward<P>(p)).make_real(); }
            
            bool compare_lexical(dotpath const&) const;   /// compare ::realpath(…) values using std::strcmp(…) --
                                                          /// this fails with nonexistant dotpaths
            
            bool compare(dotpath const&) const noexcept;  /// compare stringified dotpaths using fast-as-fuck dotpath::hash() --
                                                          /// this works as expected with nonexistant dotpath
            
            /// static forwarders for dotpath::compare_lexical<P, Q>(p, q) and dotpath::compare<P, Q>(p, q)
            template <typename P, typename Q> inline
            static bool compare_lexical(P&& p, Q&& q) {
                dotpath lhs(std::forward<P>(p));
                dotpath rhs(std::forward<Q>(q));
                if (!lhs.exists() || !rhs.exists()) { return false; }
                return lhs.compare_lexical(rhs);
            }
            
            template <typename P, typename Q> inline
            static bool compare(P&& p, Q&& q) {
                return dotpath(std::forward<P>(p)).compare(dotpath(std::forward<Q>(q)));
            }
            
            /// equality-test operators use dotpath::compare(…) and therefore dotpath::hash()
            bool operator==(dotpath const&) const;
            bool operator!=(dotpath const&) const;
            
            /// self-explanatory interrogatives
            bool exists() const;
            
            /// Static forwarders for the aforementioned interrogatives
            template <typename P> inline
            static bool exists(P&& p) { return dotpath(std::forward<P>(p)).exists(); }
            
            /// Convenience funcs for running a std::regex against the dotpath in question:
            /// match(), search() and replace() hand respectively straight off to std::regex_match,
            /// std::regex_search(), and std::regex_replace, using the stringified dotpath.
            /// As one would expect, calls to the former two return booleans whereas
            /// both replace() overloads yield new dotpath objects.
            bool match(std::regex const& pattern,           bool case_sensitive=false) const;
            bool search(std::regex const& pattern,          bool case_sensitive=false) const;
            dotpath replace(std::regex const& pattern,      char const* replacement,
                                                            bool case_sensitive=false) const;
            dotpath replace(std::regex const& pattern,      std::string const& replacement,
                                                            bool case_sensitive=false) const;
            
            /// static forwarder for dotpath::match<P>(p, pattern)
            template <typename P> inline
            static bool match(P&& p, std::regex&& pattern,  bool case_sensitive=false) {
                return dotpath(std::forward<P>(p)).match(
                    std::forward<std::regex>(pattern), case_sensitive);
            }
            
            /// static forwarder for dotpath::search<P>(p, pattern)
            template <typename P> inline
            static bool search(P&& p, std::regex&& pattern, bool case_sensitive=false) {
                return dotpath(std::forward<P>(p)).search(
                    std::forward<std::regex>(pattern), case_sensitive);
            }
            
            /// static forwarder for dotpath::replace<P, S>(p, pattern, s)
            template <typename P, typename S> inline
            static dotpath replace(P&& p, std::regex&& pattern, S&& s, bool case_sensitive=false) {
                return dotpath(std::forward<P>(p)).replace(
                    std::forward<std::regex>(pattern), std::forward<S>(s), case_sensitive);
            }
            
            /// get the basename -- i.e. for dotpath yo.dogg.iheardyoulike.basenames:dogg
            /// ... the basename returned is "basenames:dogg"
            std::string basename() const;
            
            /// Static forwarder for dotpath::basename<P>(p)
            template <typename P> inline
            static std::string basename(P&& p) {
                return dotpath(std::forward<P>(p)).basename();
            }
            
            /// Attempt to return the string extention (WITHOUT THE LEADING ":")
            std::string extension() const;
            std::string extensions() const;
            
            /// Static forwarder for dotpath::extension<P>(p) and dotpath::extensions<P>(p)
            template <typename P> inline
            static std::string extension(P&& p) {
                return dotpath(std::forward<P>(p)).extension();
            }
            
            template <typename P> inline
            static std::string extensions(P&& p) {
                return dotpath(std::forward<P>(p)).extensions();
            }
            
            /// Return a new dotpath with the extension stripped off, e.g.
            ///  dotpath("yo.dogg:tar:gz").strip_extension() == dotpath("yo.dogg:tar");
            /// dotpath("yo.dogg:tar:gz").strip_extensions() == dotpath("yo.dogg");
            dotpath strip_extension() const;
            dotpath strip_extensions() const;
            
            /// Static forwarder for dotpath::strip_extension<P>(p) and dotpath::strip_extensions<P>(p)
            template <typename P> inline
            static dotpath strip_extension(P&& p) {
                return dotpath(std::forward<P>(p)).strip_extension();
            }
            
            template <typename P> inline
            static dotpath strip_extensions(P&& p) {
                return dotpath(std::forward<P>(p)).strip_extensions();
            }
            
            /// Convenience method to get the result of dotpath::extensions(),
            /// split on dotpath::extsep, as a vector of strings.
            detail::stringvec_t split_extensions() const;
            
            /// Static forwarder for dotpath::split_extensions<P>(p)
            template <typename P> inline
            static detail::stringvec_t split_extensions(P&& p) {
                return dotpath(std::forward<P>(p)).split_extensions();
            }
            
            /// Get back the parent dotpath (also known as the 'dirname' if you are
            /// a fan of the Python os.path module, which meh I could take or leave)
            dotpath parent() const;
            dotpath dirname() const;
            
            /// Static forwarder for dotpath::parent<P>(p) and dotpath::dirname<P>(p)
            template <typename P> inline
            static dotpath parent(P&& p) {
                return dotpath(std::forward<P>(p)).parent();
            }
            
            template <typename P> inline
            static dotpath dirname(P&& p) {
                return dotpath(std::forward<P>(p)).dirname();
            }
            
            /// join a dotpath with another trailing dotpath fragment, creating a new path:
            dotpath join(dotpath const& other) const;
            
            /// join a path in place with another trailing path fragment and return it:
            dotpath& adjoin(dotpath const& other);
            
            /// operator overloads to join dotpaths with slashes -- you can be like this:
            ///     path p = "yo.dogg";
            ///     path q = p / "i-heard";
            ///     path r = q / "you-like";
            ///     path s = r / "to-join-paths";
            dotpath operator/(dotpath const& other) const;
            dotpath operator/(char const* other) const;
            dotpath operator/(std::string const& other) const;
            
            /// analogous join-in-place slash-equals operators:
            dotpath& operator/=(dotpath const&);
            dotpath& operator/=(char const*);
            dotpath& operator/=(std::string const&);
            
            /// Static forwarder for dotpath::join<P, Q>(p, q) --
            /// sometimes you want to just join stuff mainually like:
            ///     dotpath p = dotpath::join(p, "something.else");
            /// ... for aesthetic purposes (versus the operator overloads), etc
            template <typename P, typename Q> inline
            static dotpath join(P&& one, Q&& theother) {
                return dotpath(std::forward<P>(one)) / std::forward<Q>(theother);
            }
            
            /// Simple string append for the trailing dotpath segment:
            dotpath append(std::string const& appendix) const;
            
            /// Simple string append-in-place for the trailing path segment:
            dotpath& extend(std::string const& appendix);
            
            /// operator overloads for bog-standard string-appending -- like so:
            ///     path p = "yo.dogg";
            ///     path q = p + "_i_heard";
            ///     path r = q + "_you_dont_necessarily_like";
            ///     path s = r + "_segment_based_append_operations";
            dotpath operator+(dotpath const& other) const;
            dotpath operator+(char const* other) const;
            dotpath operator+(std::string const& other) const;
            
            /// analogous append-in-place plus-equals operators:
            dotpath& operator+=(dotpath const&);
            dotpath& operator+=(char const*);
            dotpath& operator+=(std::string const&);
            
            /// Static forwarder for dotpath::append<P, Q>(p, q) --
            /// you *get* this by now, rite? It's just like some shorthand for
            ///     dotpath p = dotpath::append(p, ".newFileExt"); /// OR WHATEVER DUDE SRSLY UGH
            template <typename P, typename Q> inline
            static dotpath append(P&& one, Q&& theother) {
                return dotpath(std::forward<P>(one)) + std::forward<Q>(theother);
            }
            
            /// std::vector<…>-style subscripting for per-segment access to the dotpath’s tokens:
            std::string&       operator[](size_type idx);
            std::string const& operator[](size_type idx) const;
            std::string&               at(size_type idx);
            std::string const&         at(size_type idx) const;
            
            /// Convenience front()/back() segment access:
            std::string&            front();
            std::string const&      front() const;
            std::string&             back();
            std::string const&       back() const;
            
            /// Stringify the dotpath:
            std::string str() const;
            
            /// Convenience functions to get C-style strings, a la std::string's API:
            char const* c_str() const;
            char const* data() const;
            
            /// Dotpath rank value per extension
            size_type rank(std::string const&) const;
            size_type rank() const;
            
            /// Static forwarders for dotpath::rank<P, S>(p, s) and dotpath::rank<P>(p)
            template <typename P, typename S> inline
            static size_type rank(P&& p, S&& s) {
                return dotpath(std::forward<P>(p)).rank(std::forward<S>(s));
            }
            
            template <typename P> inline
            static size_type rank(P&& p) {
                return dotpath(std::forward<P>(p)).rank();
            }
            
            /// Conversion operators -- in theory you can pass your dotpaths to functions
            /// expecting either std::strings or const char*s with these...
            operator std::string() const;
            operator char const*() const;
            
            /// less-than operator -- allows the use of filesystem::dotpath in e.g. std::map
            bool operator<(dotpath const&) const noexcept;
            bool operator>(dotpath const&) const noexcept;
            
            /// Set and tokenize the dotpath using a std::string (mainly used internally)
            void set(std::string const&);
            
            /// ... and here, we have the requisite assign operators
            dotpath& operator=(std::string const&);
            dotpath& operator=(char const*);
            dotpath& operator=(dotpath const&);
            dotpath& operator=(dotpath&&) noexcept;
            dotpath& operator=(detail::stringvec_t const&);
            dotpath& operator=(detail::stringvec_t&&) noexcept;
            dotpath& operator=(detail::stringlist_t);
            
            /// Stringify the path to an ostream
            friend std::ostream& operator<<(std::ostream&, dotpath const&);
            
            /// calculate the hash value for the dotpath
            size_type hash() const noexcept;
            
            /// Static forwarder for the hash function
            template <typename P> inline
            static dotpath::size_type hash(P&& p) {
                return dotpath(std::forward<P>(p)).hash();
            }
            
            /// no-except member swap
            void swap(dotpath&) noexcept;
            
            /// dotpath component vector
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
    void swap(filesystem::dotpath&, filesystem::dotpath&) noexcept;
    
    /// std::hash specialization for filesystem::dotpath
    /// ... following the recipe found here:
    ///     http://en.cppreference.com/w/cpp/utility/hash#Examples
    
    template <>
    struct hash<filesystem::dotpath> {
        
        typedef filesystem::dotpath argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const&) const;
        
    };
    
} /* namespace std */

#endif /// LIBIMREAD_EXT_FILESYSTEM_DOTPATH_H_