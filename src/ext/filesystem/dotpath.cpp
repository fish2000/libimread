/// Copyright 2014-2018 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <cerrno>
#include <thread>
#include <numeric>
#include <algorithm>
#include <type_traits>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/dotpath.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/pystring.hh>
#include <libimread/errors.hh>
#include <libimread/rehash.hh>

namespace filesystem {
    
    namespace detail {
        
        using rehasher_t = hash::rehasher<std::string>;
        
        static constexpr std::regex::flag_type regex_flags          = std::regex::extended;
        static constexpr std::regex::flag_type regex_flags_icase    = std::regex::extended | std::regex::icase;
        
        static const std::string extsepstring(1, dotpath::extsep);
        static const std::string sepstring(1, dotpath::sep);
        static const std::string nulstring("");
        
    } /* namespace detail */
    
    constexpr dotpath::character_type dotpath::sep;
    constexpr dotpath::character_type dotpath::extsep;
    constexpr dotpath::character_type dotpath::pathsep;
    
    dotpath::dotpath() {}
    dotpath::dotpath(bool abs)
        :m_absolute(abs)
        {}
    
    dotpath::dotpath(dotpath const& dp)
        :m_absolute(dp.m_absolute)
        ,m_path(dp.m_path)
        {}
    
    dotpath::dotpath(dotpath&& dp) noexcept
        :m_absolute(dp.m_absolute)
        ,m_path(std::move(dp.m_path))
        {}
    
    dotpath::dotpath(path const& p)
        :m_absolute(p.is_absolute())
        ,m_path(p.components())
        {
            m_path.back() = std::regex_replace(m_path.back(),
                                               std::regex("\\" + std::string(1, path::extsep)),
                                               detail::extsepstring);
        }
    
    dotpath::dotpath(path&& p) noexcept
        :m_absolute(p.is_absolute())
        ,m_path(p.components())
        {
            m_path.back() = std::regex_replace(m_path.back(),
                                               std::regex("\\" + std::string(1, path::extsep)),
                                               detail::extsepstring);
        }
    
    dotpath::dotpath(char* st)              { set(st); }
    dotpath::dotpath(char const* st)        { set(st); }
    dotpath::dotpath(std::string const& st) { set(st); }
    
    dotpath::dotpath(detail::stringvec_t const& vec, bool absolute)
        :m_absolute(absolute)
        ,m_path(vec)
        {}
    
    dotpath::dotpath(detail::stringvec_t&& vec, bool absolute) noexcept
        :m_absolute(absolute)
        ,m_path(std::move(vec))
        {}
    
    dotpath::dotpath(detail::stringlist_t list)
        :m_path(list)
        {}
    
    dotpath::~dotpath() {}
    
    dotpath::size_type dotpath::size() const { return static_cast<dotpath::size_type>(m_path.size()); }
    dotpath::size_type dotpath::length() const { return static_cast<dotpath::size_type>(str().size()); }
    bool dotpath::is_absolute() const { return m_absolute; }
    bool dotpath::empty() const       { return m_path.empty(); }
    
    bool dotpath::match(std::regex const& pattern, bool case_sensitive) const {
        return std::regex_match(str(), pattern);
    }
    
    bool dotpath::search(std::regex const& pattern, bool case_sensitive) const {
        return std::regex_search(str(), pattern);
    }
    
    dotpath dotpath::replace(std::regex const& pattern, char const* replacement, bool case_sensitive) const {
        return dotpath(std::regex_replace(str(), pattern, replacement));
    }
    dotpath dotpath::replace(std::regex const& pattern, std::string const& replacement, bool case_sensitive) const {
        return dotpath(std::regex_replace(str(), pattern, replacement));
    }
    
    dotpath dotpath::make_absolute() const {
        dotpath out(true);
        out.m_path.reserve(size());
        std::copy(m_path.begin(),
                  m_path.end(),
                  std::back_inserter(out.m_path));
        return out;
    }
    
    dotpath dotpath::make_real() const {
        /// same as dotpath::make_absolute(), minus the m_absolute precheck:
        dotpath out(true);
        out.m_path.reserve(size());
        std::copy(m_path.begin(),
                  m_path.end(),
                  std::back_inserter(out.m_path));
        return out;
    }

    bool dotpath::compare_lexical(dotpath const& other) const {
        if (!exists() || !other.exists()) { return false; }
        std::string self_string = str(),
                    other_string = other.str();
        return bool(std::strcmp(self_string.c_str(), other_string.c_str()) == 0);
    }
    
    bool dotpath::compare(dotpath const& other) const noexcept {
        return bool(hash() == other.hash());
    }
    
    bool dotpath::operator==(dotpath const& other) const { return bool(hash() == other.hash()); }
    bool dotpath::operator!=(dotpath const& other) const { return bool(hash() != other.hash()); }
    
    bool dotpath::exists() const {
        // return ::access(c_str(), F_OK) != -1;
        return true;
    }
    
    std::string dotpath::basename() const {
        return m_path.empty() ? "" : m_path.back();
    }
    
    std::string dotpath::extension() const {
        if (m_path.empty()) { return ""; }
        std::string const& last = m_path.back();
        size_type pos = last.find_last_of(dotpath::extsep);
        if (pos == std::string::npos) { return ""; }
        return last.substr(pos+1);
    }
    
    std::string dotpath::extensions() const {
        if (m_path.empty()) { return ""; }
        std::string const& last = m_path.back();
        size_type pos = last.find_first_of(dotpath::extsep);
        if (pos == std::string::npos) { return ""; }
        return last.substr(pos+1);
    }
    
    dotpath dotpath::strip_extension() const {
        if (m_path.empty()) { return dotpath(); }
        dotpath result(m_path, m_absolute);
        std::string const& ext(extension());
        std::string& back(result.m_path.back());
        back = back.substr(0, back.size() - (ext.size() + 1));
        return result;
    }
    
    dotpath dotpath::strip_extensions() const {
        if (m_path.empty()) { return dotpath(); }
        dotpath result(m_path, m_absolute);
        std::string const& ext(extensions());
        std::string& back(result.m_path.back());
        back = back.substr(0, back.size() - (ext.size() + 1));
        return result;
    }
    
    detail::stringvec_t dotpath::split_extensions() const {
        detail::stringvec_t out;
        pystring::split(extensions(), out, detail::extsepstring);
        return out;
    }
    
    dotpath dotpath::parent() const {
        dotpath result;
        if (m_path.empty()) {
            if (m_absolute) {
                imread_raise(FileSystemError,
                    "dotpath::parent() makes no sense for empty absolute dotpaths");
            } else {
                return result;
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
    dotpath dotpath::dirname() const { return parent(); }
    
    dotpath dotpath::join(dotpath const& other) const {
        dotpath result(m_path, m_absolute);
        result.m_path.reserve(m_path.size() + other.m_path.size());
        std::copy(other.m_path.begin(),
                  other.m_path.end(),
                  std::back_inserter(result.m_path));
        return result;
    }
    
    dotpath& dotpath::adjoin(dotpath const& other) {
        m_path.reserve(m_path.size() + other.m_path.size());
        std::copy(other.m_path.begin(),
                  other.m_path.end(),
                  std::back_inserter(m_path));
        return *this;
    }
    
    dotpath dotpath::operator/(dotpath const& other) const     { return join(other); }
    dotpath dotpath::operator/(char const* other) const        { return join(dotpath(other)); }
    dotpath dotpath::operator/(std::string const& other) const { return join(dotpath(other)); }
    
    dotpath& dotpath::operator/=(dotpath const& other)     { return adjoin(other); }
    dotpath& dotpath::operator/=(char const* other)        { return adjoin(dotpath(other)); }
    dotpath& dotpath::operator/=(std::string const& other) { return adjoin(dotpath(other)); }
    
    dotpath dotpath::append(std::string const& appendix) const {
        dotpath out(m_path.empty() ? detail::stringvec_t{ "" } : m_path, m_absolute);
        out.m_path.back().append(appendix);
        return out;
    }
    
    dotpath& dotpath::extend(std::string const& appendix) {
        if (appendix.empty()) {
            return *this;
        }
        if (m_path.empty()) {
            m_path = detail::stringvec_t{ "" };
        }
        m_path.back().append(appendix);
        return *this;
    }
    
    std::string&       dotpath::operator[](size_type idx)       { return m_path[idx]; }
    std::string const& dotpath::operator[](size_type idx) const { return m_path[idx]; }
    std::string&               dotpath::at(size_type idx)       { return m_path.at(idx); }
    std::string const&         dotpath::at(size_type idx) const { return m_path.at(idx); }
    
    std::string&            dotpath::front()                    { return m_path.front(); }
    std::string const&      dotpath::front() const              { return m_path.front(); }
    std::string&            dotpath::back()                     { return m_path.back(); }
    std::string const&      dotpath::back() const               { return m_path.back(); }
    
    dotpath dotpath::operator+(dotpath const& other) const      { return append(other.str()); }
    dotpath dotpath::operator+(char const* other) const         { return append(other); }
    dotpath dotpath::operator+(std::string const& other) const  { return append(other); }
    
    dotpath& dotpath::operator+=(dotpath const& other)          { return extend(other.str()); }
    dotpath& dotpath::operator+=(char const* other)             { return extend(other); }
    dotpath& dotpath::operator+=(std::string const& other)      { return extend(other); }
    
    std::string dotpath::str() const {
        return std::accumulate(m_path.begin(),
                               m_path.end(),
                               m_absolute ? detail::nulstring
                                          : detail::sepstring,
                           [&](std::string const& lhs,
                               std::string const& rhs) {
            return lhs + rhs + ((rhs.c_str() == m_path.back().c_str()) ? detail::nulstring
                                                                       : detail::sepstring);
        });
    }
    
    char const* dotpath::c_str() const {
        return str().c_str();
    }
    
    char const* dotpath::data() const {
        return str().data();
    }
    
    dotpath::size_type dotpath::rank(std::string const& ext) const {
        /// I can't remember from whence I stole this implementation:
        std::string thisdotpath = str();
        if (thisdotpath.size() >= ext.size() &&
            thisdotpath.compare(thisdotpath.size() - ext.size(), ext.size(), ext) == 0) {
            if (thisdotpath.size() == ext.size()) {
                return ext.size();
            }
            char ch = thisdotpath[thisdotpath.size() - ext.size() - 1];
            if (ch == '.' || ch == '_') {
                return ext.size() + 1;
            } else if (ch == '/') {
                return ext.size();
            }
        }
        return 0;
    }
    
    dotpath::size_type dotpath::rank() const {
        return dotpath::rank(dotpath::extension());
    }
    
    dotpath::operator std::string() const  { return str(); }
    dotpath::operator char const*() const  { return c_str(); }
    
    bool dotpath::operator<(dotpath const& rhs) const noexcept {
        return size() < rhs.size();
    }
    
    bool dotpath::operator>(dotpath const& rhs) const noexcept {
        return size() > rhs.size();
    }
    
    void dotpath::set(std::string const& str) {
        m_absolute = !str.empty() && str[0] == dotpath::sep;
        m_path = tokenize(str, dotpath::sep);
    }
    
    dotpath& dotpath::operator=(std::string const& str) { set(str); return *this; }
    dotpath& dotpath::operator=(char const* str)        { set(str); return *this; }
    dotpath& dotpath::operator=(dotpath const& dp) {
        if (hash() != dp.hash()) {
            dotpath(dp).swap(*this);
        }
        return *this;
    }
    
    dotpath& dotpath::operator=(dotpath&& dp) noexcept {
        if (hash() != dp.hash()) {
            m_absolute = dp.m_absolute;
            m_path = std::exchange(dp.m_path, detail::stringvec_t{});
        }
        return *this;
    }
    
    dotpath& dotpath::operator=(detail::stringvec_t const& stringvec) {
        m_path = detail::stringvec_t{};
        m_path.reserve(stringvec.size());
        std::copy(stringvec.begin(),
                  stringvec.end(),
                  std::back_inserter(m_path));
        return *this;
    }
    
    dotpath& dotpath::operator=(detail::stringvec_t&& stringvec) noexcept {
        m_path = std::move(stringvec);
        return *this;
    }
    
    dotpath& dotpath::operator=(detail::stringlist_t stringlist) {
        m_path = detail::stringvec_t(stringlist);
        return *this;
    }
    
    std::ostream& operator<<(std::ostream& os, dotpath const& dp) {
        return os << dp.str();
    }
    
    dotpath::size_type dotpath::hash() const noexcept {
        /// calculate the hash value for the dotpath
        return std::accumulate(m_path.begin(),
                               m_path.end(),
                               static_cast<dotpath::size_type>(m_absolute),
                               detail::rehasher_t());
    }
    
    void dotpath::swap(dotpath& other) noexcept {
        using std::swap;
        swap(m_absolute, other.m_absolute);
        swap(m_path,     other.m_path);
    }
    
    detail::stringvec_t dotpath::components() const {
        /// return dotpath component vector
        detail::stringvec_t out;
        out.reserve(m_path.size());
        std::copy(m_path.begin(),
                  m_path.end(),
                  std::back_inserter(out));
        return out;
    }
    
    detail::stringvec_t dotpath::tokenize(std::string const& source,
                                          dotpath::character_type const delim) {
        detail::stringvec_t tokens;
        dotpath::size_type lastPos = 0,
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
    
} /* namespace filesystem */

namespace std {
    
    template <>
    void swap(filesystem::dotpath& p0, filesystem::dotpath& p1) noexcept {
        p0.swap(p1);
    }
    
    using dotpath_hasher_t = std::hash<filesystem::dotpath>;
    using dotpath_arg_t = dotpath_hasher_t::argument_type;
    using dotpath_out_t = dotpath_hasher_t::result_type;
    
    dotpath_out_t dotpath_hasher_t::operator()(dotpath_arg_t const& dp) const {
        return static_cast<dotpath_out_t>(dp.hash());
    }
    
} /* namespace std */