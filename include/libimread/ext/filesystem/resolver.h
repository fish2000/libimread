// Copyright 2015 Wenzel Jakob. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#if !defined(LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_)
#define LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_

#include <libimread/ext/filesystem/path.h>

namespace filesystem {
    
    class resolver {
        
        public:
            using iterator       = detail::pathvec_t::iterator;
            using const_iterator = detail::pathvec_t::const_iterator;
            
            resolver()
                { m_paths.push_back(path::getcwd()); }
            
            template <typename P,
                      typename = std::enable_if_t<
                                 std::is_constructible<path, P>::value>>
            explicit resolver(P&& p)
                { m_paths.push_back(path(std::forward<P>(p))); }
            
            explicit resolver(detail::pathlist_t list)
                :m_paths(list)
                {}
            
            std::size_t size() const        { return m_paths.size(); }
            iterator begin()                { return m_paths.begin(); }
            iterator end()                  { return m_paths.end(); }
            const_iterator begin() const    { return m_paths.begin(); }
            const_iterator end()   const    { return m_paths.end(); }
            
            void erase(iterator it)         { m_paths.erase(it); }
            void prepend(const path& path)  { m_paths.insert(m_paths.begin(), path); }
            void append(const path& path)   { m_paths.push_back(path); }
            
            path resolve(const path& value) const {
                if (m_paths.empty()) { return path(); }
                for (const_iterator it = m_paths.begin(); it != m_paths.end(); ++it) {
                    path combined = *it / value;
                    if (combined.exists()) { return combined; }
                }
                return path();
            }
            
            detail::pathvec_t resolve_all(const path& value) const {
                detail::pathvec_t out;
                if (m_paths.empty()) { return out; }
                std::copy_if(m_paths.begin(), m_paths.end(),
                             std::back_inserter(out),
                             [&](const path& p) {
                    return (p/value).exists();
                });
                return out;
            }
            
        private:
            detail::pathvec_t m_paths;
    
    };
    
}; /* namespace filesystem */

#endif /// LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_
