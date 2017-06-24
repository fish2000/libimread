/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <numeric>
#include <libimread/ext/filesystem/resolver.h>

namespace filesystem {
    
    resolver::resolver()
        :m_paths{ path::getcwd() }
        {}
    
    resolver::resolver(detail::pathvec_t const& paths)
        :m_paths(paths)
        {
            m_paths.erase(
                std::remove_if(m_paths.begin(),
                               m_paths.end(),
                            [](path const& p) { return p == path(); }), m_paths.end());
        }
    
    resolver::resolver(detail::stringvec_t const& strings)
        {
            m_paths.reserve(strings.size());
            std::transform(strings.begin(),
                           strings.end(),
                           std::back_inserter(m_paths),
                        [](std::string const& s) { return path(s); });
            m_paths.erase(
                std::remove_if(m_paths.begin(),
                               m_paths.end(),
                            [](path const& p) { return p == path(); }), m_paths.end());
        }
    
    resolver::resolver(detail::pathlist_t list)
        :m_paths(list)
        {
            m_paths.erase(
                std::remove_if(m_paths.begin(),
                               m_paths.end(),
                            [](path const& p) { return p == path(); }), m_paths.end());
        }
    
    resolver::~resolver() {}
    
    resolver resolver::system()                         { return resolver(path::system()); }
    
    resolver::size_type resolver::size() const          { return m_paths.size(); }
    resolver::iterator resolver::begin()                { return m_paths.begin(); }
    resolver::iterator resolver::end()                  { return m_paths.end(); }
    resolver::const_iterator resolver::begin() const    { return m_paths.begin(); }
    resolver::const_iterator resolver::end() const      { return m_paths.end(); }
    
    void resolver::erase(resolver::iterator it)         { m_paths.erase(it); }
    void resolver::prepend(path const& p)               { m_paths.insert(m_paths.begin(), p); }
    void resolver::append(path const& p)                { m_paths.push_back(p); }
    
    path resolver::resolve_impl(path const& value) const {
        if (m_paths.empty()) { return path(); }
        for (const_iterator it = m_paths.begin(); it != m_paths.end(); ++it) {
            path combined = *it / value;
            if (combined.exists()) { return combined; }
        }
        return path();
    }
    
    detail::pathvec_t resolver::resolve_all_impl(path const& value) const {
        detail::pathvec_t out;
        if (m_paths.empty()) { return out; }
        std::copy_if(m_paths.begin(),
                     m_paths.end(),
                     std::back_inserter(out),
            [&value](path const& p) { return (p/value).exists(); });
        return out;
    }
    
    std::string resolver::to_string(std::string const& separator,
                                    std::string const& initial) const {
        return std::accumulate(m_paths.begin(),
                               m_paths.end(),
                               initial,
                           [&](path const& lhs,
                               path const& rhs) {
            return lhs.str() + rhs.str() + (rhs == m_paths.back() ? std::string("")
                                                                  : separator);
        });
    }
    
    std::ostream& operator<<(std::ostream& os, resolver const& paths) {
        return os << paths.to_string();
    }
    
}