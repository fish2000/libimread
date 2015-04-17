// Copyright 2015 Wenzel Jakob. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#if !defined(__FILESYSTEM_PATH_H)
#define __FILESYSTEM_PATH_H

#include <string>
#include <vector>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

namespace filesystem {
    
    /// forward declaration for these next few prototypes/templates
    class path;
    
    /// Deleter structure to close a directory handle
    template <typename D>
    struct dircloser {
        constexpr dircloser() noexcept = default;
        template <typename U> dircloser(const dircloser<U>&) noexcept {};
        void operator()(D *dirhandle) { if (dirhandle) closedir(dirhandle); }
    };
    
    using directory = std::unique_ptr<typename std::decay<DIR>::type, dircloser<DIR>>;
    
    directory ddopen(const char *c);
    directory ddopen(const std::string &s);
    directory ddopen(const path &p);
    
    /**
     * \brief Simple class for manipulating paths on Linux/Windows/Mac OS
     *
     * This class is just a temporary workaround to avoid the heavy boost
     * dependency until boost::filesystem is integrated into the standard template
     * library at some point in the future.
     */
    class path {
        
        public:
            enum path_type {
                windows_path = 0,
                posix_path = 1,
                native_path = posix_path
            };
            
            path()
                :m_type(native_path)
                ,m_absolute(false)
                {}
            
            path(const path &path)
                :m_type(path.m_type)
                ,m_path(path.m_path)
                ,m_absolute(path.m_absolute)
                {}
            
            path(path &&path)
                :m_type(path.m_type)
                ,m_path(std::move(path.m_path))
                ,m_absolute(path.m_absolute)
                {}
            
            path(const char *string) { set(string); }
            path(const std::string &string) { set(string); }
            
            inline std::size_t size() const { return m_path.size(); }
            inline bool empty() const { return m_path.empty(); }
            inline bool is_absolute() const { return m_absolute; }
            
            path make_absolute() const {
                char temp[PATH_MAX];
                if (realpath(c_str(), temp) == NULL) {
                    throw std::runtime_error("Internal error in realpath(): " + std::string(strerror(errno)));
                }
                return path(temp);
            }
            
            template <typename P> inline
            static path absolute(P p) { return path(p).make_absolute(); }
            
            bool exists() const {
                struct stat sb;
                return stat(c_str(), &sb) == 0;
            }
            
            bool is_directory() const {
                struct stat sb;
                if (stat(c_str(), &sb)) { return false; }
                return S_ISDIR(sb.st_mode);
            }
            
            bool match(const std::regex &pattern,           bool case_sensitive=false);
            bool search(const std::regex &pattern,          bool case_sensitive=false);
            
            template <typename P> inline
            static bool match(P p, const std::regex &pattern, bool case_sensitive=false) {
                return path(p).match(pattern);
            }
            
            template <typename P> inline
            static bool search(P p, const std::regex &pattern, bool case_sensitive=false) {
                return path(p).search(pattern);
            }
            
            std::vector<path> list(                                                         bool full_paths=false);
            std::vector<path> list(const char *pattern,                                     bool full_paths=false);
            std::vector<path> list(const std::string &pattern,                              bool full_paths=false);
            std::vector<path> list(const std::regex &pattern, bool case_sensitive=false,    bool full_paths=false);
            
            template <typename P, typename G> inline
            static std::vector<path> list(P p, G g) { return path(p).list(g); }
            
            bool is_file() const {
                struct stat sb;
                if (stat(c_str(), &sb)) { return false; }
                return S_ISREG(sb.st_mode);
            }
            
            std::string extension() const {
                if (empty()) { return ""; }
                const std::string &last = m_path[m_path.size()-1];
                std::size_t pos = last.find_last_of(".");
                if (pos == std::string::npos) { return ""; }
                return last.substr(pos+1);
            }
            
            path parent_path() const {
                path result;
                result.m_absolute = m_absolute;
                
                if (m_path.empty()) {
                    if (!m_absolute) { result.m_path.push_back(".."); }
                } else {
                    std::size_t until = m_path.size() - 1;
                    for (std::size_t i = 0; i < until; ++i) {
                        result.m_path.push_back(m_path[i]);
                    }
                }
                return result;
            }
            
            path join(const path &other) const {
                if (other.m_absolute) {
                    throw std::runtime_error("path::operator/(): expected a relative path!");
                }
                if (other.m_type != other.m_type) {
                    throw std::runtime_error("path::operator/(): expected a path of the same type!");
                }
                
                path result(*this);
                
                for (std::size_t i = 0; i < other.m_path.size(); ++i) {
                    result.m_path.push_back(other.m_path[i]);
                }
                
                return result;
            }
            
            path operator/(const path &other) const { return join(other); }
            path operator/(const char *other) const { return join(path(other)); }
            path operator/(const std::string &other) const { return join(path(other)); }
            
            template <typename P, typename Q> inline
            static path join(const P one, const Q theother) { return path(one)/path(theother); }
            
            std::string str(path_type type = native_path) const {
                std::ostringstream oss;
                
                if (m_type == posix_path && m_absolute)
                    oss << "/";
                
                for (std::size_t i = 0; i < m_path.size(); ++i) {
                    oss << m_path[i];
                    if (i + 1 < m_path.size()) {
                        if (type == posix_path) { oss << '/'; }
                        else { oss << '\\'; }
                    }
                }
                
                return oss.str();
            }
            
            inline const char *c_str() const { return str().c_str(); }
            
            static path getcwd() {
                char temp[PATH_MAX];
                if (::getcwd(temp, PATH_MAX) == NULL) {
                    throw std::runtime_error("Internal error in getcwd(): " + std::string(strerror(errno)));
                }
                return path(temp);
            }
            
            /* static path cwd() __attribute__ ((weakref("getcwd"))); */
            
            static path cwd()               { return path::getcwd(); }
            
            operator std::string()          { return str(); }
            operator const char*()          { return c_str(); }
            
            void set(const std::string &str, path_type type = native_path) {
                m_type = type;
                if (type == windows_path) {
                    m_path = tokenize(str, "/\\");
                    m_absolute = str.size() >= 2 && std::isalpha(str[0]) && str[1] == ':';
                } else {
                    m_path = tokenize(str, "/");
                    m_absolute = !str.empty() && str[0] == '/';
                }
            }
            
            path &operator=(const std::string &str) { set(str); return *this; }
            path &operator=(const path &path) {
                m_type = path.m_type;
                m_path = path.m_path;
                m_absolute = path.m_absolute;
                return *this;
            }
            path &operator=(path &&path) {
                if (this != &path) {
                    m_type = path.m_type;
                    m_path = std::move(path.m_path);
                    m_absolute = path.m_absolute;
                }
                return *this;
            }
            
            friend std::ostream &operator<<(std::ostream &os, const path &path) {
                os << path.str();
                return os;
            }
            
        protected:
            static std::vector<std::string> tokenize(const std::string &string, const std::string &delim) {
                std::string::size_type lastPos = 0, pos = string.find_first_of(delim, lastPos);
                std::vector<std::string> tokens;
            
                while (lastPos != std::string::npos) {
                    if (pos != lastPos)
                        tokens.push_back(string.substr(lastPos, pos - lastPos));
                    lastPos = pos;
                    if (lastPos == std::string::npos || lastPos + 1 == string.length())
                        break;
                    pos = string.find_first_of(delim, ++lastPos);
                }
                
                return tokens;
            }
            
        protected:
            path_type m_type;
            std::vector<std::string> m_path;
            bool m_absolute;
    };
    
    /// change directory temporarily with RAII
    struct switchdir {
        
        switchdir(path newdir)
            :olddir(path::cwd().str())
            { chdir(newdir.c_str()); }
        
        ~switchdir() { chdir(olddir.c_str()); }
        
        std::string olddir;
    
    };
    
    
}; /* namespace filesystem */

#endif /* __FILESYSTEM_PATH_H */
