// Copyright 2015 Wenzel Jakob. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#if !defined(LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_)
#define LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_

#include <libimread/ext/filesystem/path.h>

namespace filesystem {

    class resolver {
        
        public:
            typedef detail::pathvec_t::iterator iterator;
            typedef detail::pathvec_t::const_iterator const_iterator;
            
            resolver()
                { m_paths.push_back(path::getcwd()); }
            
            std::size_t size() const        { return m_paths.size(); }
            iterator begin()                { return m_paths.begin(); }
            iterator end()                  { return m_paths.end(); }
            const_iterator begin() const    { return m_paths.begin(); }
            const_iterator end()   const    { return m_paths.end(); }
            
            void erase(iterator it)         { m_paths.erase(it); }
            void prepend(const path& path)  { m_paths.insert(m_paths.begin(), path); }
            void append(const path& path)   { m_paths.push_back(path); }
            
            path resolve(const path& value) const {
                for (const_iterator it = m_paths.begin(); it != m_paths.end(); ++it) {
                    path combined = *it / value;
                    if (combined.exists()) { return combined; }
                }
                return path();
            }
            
        private:
            detail::pathvec_t m_paths;
    
    };

}; /* namespace filesystem */

#endif /// LIBIMREAD_EXT_FILESYSTEM_RESOLVER_H_
