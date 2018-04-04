/// Copyright 2014-2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#if !defined(LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_)
#define LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_

#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    class resolver {
        
        public:
            using size_type         = detail::pathvec_t::size_type;
            using iterator          = detail::pathvec_t::iterator;
            using const_iterator    = detail::pathvec_t::const_iterator;
            
            resolver();
            resolver(resolver const&) = default;
            resolver(resolver&&) noexcept = default;
            
            template <typename P,
                      typename = std::enable_if_t<
                                 std::is_constructible<path, P>::value &&
                                !std::is_same<detail::stringvec_t, P>::value>>
            explicit resolver(P&& p)
                :m_paths{ path(std::forward<P>(p)) }
                {}
            
            explicit resolver(detail::pathvec_t const& paths);
            explicit resolver(detail::stringvec_t const& strings);
            explicit resolver(detail::pathlist_t list);
            virtual ~resolver();
            
            static resolver system();
            
            size_type size() const;
            iterator begin();
            iterator end();
            const_iterator begin() const;
            const_iterator end() const;
            
            void erase(iterator);
            void prepend(path const&);
            void append(path const&);
            
            path resolve_impl(path const&) const;
            detail::pathvec_t resolve_all_impl(path const&) const;
            
            template <typename P> inline
            path resolve(P&& p) const {
                return resolve_impl(path(std::forward<P>(p)));
            }
            
            template <typename P> inline
            bool contains(P&& p) const {
                return resolve_impl(path(std::forward<P>(p))) != path();
            }
            
            template <typename P> inline
            detail::pathvec_t resolve_all(P&& p) const {
                return resolve_all_impl(path(std::forward<P>(p)));
            }
            
            std::string to_string(std::string const& separator = std::string(1, path::sep),
                                  std::string const& initial = "") const;
            
            friend std::ostream& operator<<(std::ostream& os, resolver const& paths);
            
        protected:
            detail::pathvec_t m_paths;
    
    };
    
} /* namespace filesystem */

#endif /// LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_
