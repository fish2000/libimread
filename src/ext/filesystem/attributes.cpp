
#if defined(__gnu_linux__) || \
   (defined(__FreeBSD_kernel__) && defined(__GLIBC__) && !defined(__FreeBSD__)) || \
    defined(__CYGWIN32__)
#define PXALINUX 1
#endif

#if defined(__FreeBSD__) || defined(PXALINUX) || defined(__APPLE__)

#include <sys/types.h>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#if defined(__FreeBSD__)
#include <sys/extattr.h>
#include <sys/uio.h>

#elif defined(PXALINUX)
#include <sys/xattr.h>

#elif defined(__APPLE__)
#include <sys/xattr.h>

#else
#error "Unknown system - can't compile"
#endif

#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/attributes.h>

namespace filesystem {
    
    namespace attribute {
        
        bool operator&(ns lhs, ns rhs) {
            return (std::size_t)lhs & (std::size_t)rhs;
        }
        
        bool operator&(flags lhs, flags rhs) {
            return (std::size_t)lhs & (std::size_t)rhs;
        }
        
        enum struct ns : std::size_t {
            user        = 0
        };
        
        enum struct flags : std::size_t {
            none        = 0,
            nofollow    = 1,
            create      = 2,
            replace     = 4
        };
        
        namespace detail {
            
            inline attrbuf_t attrbuf(std::size_t size) {
                return std::make_unique<char[]>(size);
            }
            
            std::string sysname(std::string const& pname, attribute::ns domain) {
                if (domain != attribute::ns::user) {
                    errno = EINVAL;
                    return detail::nullstring;
                }
                return std::string(detail::userstring + pname);
            }
            
            std::string pxaname(std::string const& sname, attribute::ns domain) {
                if (!userstring.empty() && sname.find(userstring) != 0) {
                    errno = EINVAL;
                    return detail::nullstring;
                }
                return std::string(sname.substr(userstring.length()));
            }
            
        } /// namespace detail
        
        std::string get(std::string const& pth,
                        std::string const& name_,
                        attribute::flags options,
                        attribute::ns domain) {
            std::string name = detail::sysname(name_);
            if (name == detail::nullstring) { return detail::nullstring; }
            
            int status = -1;
            detail::attrbuf_t attrbuffer;
            
            #if defined(__FreeBSD__)
                if (options & attribute::flags::nofollow) {
                    status = ::extattr_get_link(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), 0, 0);
                } else {
                    status = ::extattr_get_file(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), 0, 0);
                }
                if (status < 0) { return detail::nullstring; }
                
                /// Don't want to deal with possible status = 0:
                attrbuffer = detail::attrbuf(status + 1);
                
                if (options & attribute::flags::nofollow) {
                    status = ::extattr_get_link(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), attrbuffer.get(), status);
                } else {
                    status = ::extattr_get_file(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), attrbuffer.get(), status);
                }
            
            #elif defined(PXALINUX)
                if (options & attribute::flags::nofollow) {
                    status = ::lgetxattr(pth.c_str(), name.c_str(), 0, 0);
                } else {
                    status = ::getxattr(pth.c_str(), name.c_str(), 0, 0);
                }
                if (status < 0) { return detail::nullstring; }
                
                /// Don't want to deal with possible status = 0:
                attrbuffer = detail::attrbuf(status + 1);
                
                if (options & attribute::flags::nofollow) {
                    status = ::lgetxattr(pth.c_str(), name.c_str(), attrbuffer.get(), status);
                } else {
                    status = ::getxattr(pth.c_str(), name.c_str(), attrbuffer.get(), status);
                }
            
            #elif defined(__APPLE__)
                if (options & attribute::flags::nofollow) {
                    status = ::getxattr(pth.c_str(), name.c_str(), 0, 0, 0, XATTR_NOFOLLOW);
                } else {
                    status = ::getxattr(pth.c_str(), name.c_str(), 0, 0, 0, 0);
                }
                if (status < 0) { return detail::nullstring; }
                
                /// Don't want to deal with possible status = 0:
                attrbuffer = detail::attrbuf(status + 1);
                
                if (options & attribute::flags::nofollow) {
                    status = ::getxattr(pth.c_str(), name.c_str(), attrbuffer.get(), status, 0, XATTR_NOFOLLOW);
                } else {
                    status = ::getxattr(pth.c_str(), name.c_str(), attrbuffer.get(), status, 0, 0);
                }
            
            #endif
            
            if (attrbuffer.get()) {
                return std::string(attrbuffer.get(), status);
            }
            return detail::nullstring;
        }
        
        bool set(std::string const& pth,
                 std::string const& name_,
                 std::string const& value,
                 attribute::flags options,
                 attribute::ns domain) {
            std::string name = detail::sysname(name_);
            if (name == detail::nullstring) { return false; }
            
            int status = -1;
            
            #if defined(__FreeBSD__)
                if (options & (attribute::flags::create | attribute::flags::replace)) {
                    /// Need to test for existence:
                    bool exists = false;
                    ssize_t error_value;
                    if (options & attribute::flags::nofollow) {
                        error_value = ::extattr_get_link(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), 0, 0);
                    } else {
                        error_value = ::extattr_get_file(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), 0, 0);
                    }
                    if (error_value >= 0) {
                        exists = true;
                    }
                    if (error_value < 0 && errno != ENOATTR) {
                        return false;
                    }
                    if ((options & attribute::flags::create) && exists) {
                        errno = EEXIST;
                        return false;
                    }
                    if ((options & attribute::flags::replace) && !exists) {
                        errno = ENOATTR;
                        return false;
                    }
                }
                if (options & attribute::flags::nofollow) {
                    status = ::extattr_set_link(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), value.c_str(), value.length());
                } else {
                    status = ::extattr_set_file(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str(), value.c_str(), value.length());
                }
            
            #elif defined(PXALINUX)
                int callopts = 0;
                if (options & attribute::flags::create) {
                    callopts = XATTR_CREATE;
                } else if (options & attribute::flags::replace) {
                    callopts = XATTR_REPLACE;
                }
                if (options & attribute::flags::nofollow) {
                    status = ::lsetxattr(pth.c_str(), name.c_str(), value.c_str(), value.length(), callopts);
                } else {
                    status = ::setxattr(pth.c_str(), name.c_str(), value.c_str(), value.length(), callopts);
                }
            
            #elif defined(__APPLE__)
                int callopts = 0;
                if (options & attribute::flags::create) {
                    callopts = XATTR_CREATE;
                } else if (options & attribute::flags::replace) {
                    callopts = XATTR_REPLACE;
                }
                if (options & attribute::flags::nofollow) {
                    status = ::setxattr(pth.c_str(), name.c_str(), value.c_str(), value.length(), 0, XATTR_NOFOLLOW | callopts);
                } else {
                    status = ::setxattr(pth.c_str(), name.c_str(), value.c_str(), value.length(), 0, callopts);
                }
            
            #endif
            
            return status >= 0;
        }
        
        bool del(std::string const& pth,
                 std::string const& name_,
                 attribute::flags options,
                 attribute::ns domain) {
            std::string name = detail::sysname(name_);
            if (name == detail::nullstring) { return false; }
            
            int status = -1;
            
            #if defined(__FreeBSD__)
                if (options & attribute::flags::nofollow) {
                    status = ::extattr_delete_link(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str());
                } else {
                    status = ::extattr_delete_file(pth.c_str(), EXTATTR_NAMESPACE_USER, name.c_str());
                }
                 
            #elif defined(PXALINUX)
                if (options & attribute::flags::nofollow) {
                    status = ::lremovexattr(pth.c_str(), name.c_str());
                } else {
                    status = ::removexattr(pth.c_str(), name.c_str());
                }
            
            #elif defined(__APPLE__)
                if (options & attribute::flags::nofollow) {
                    status = ::removexattr(pth.c_str(), name.c_str(), XATTR_NOFOLLOW);
                } else {
                    status = ::removexattr(pth.c_str(), name.c_str(), 0);
                }
            
            #endif
            
            return status >= 0;
        }
        
        detail::stringvec_t list(std::string const& pth,
                                 attribute::flags options,
                                 attribute::ns domain) {
            int status = -1;
            detail::attrbuf_t attrbuffer;
            
            #if defined(__FreeBSD__)
                if (options & attribute::flags::nofollow) {
                    status = ::extattr_list_link(pth.c_str(), EXTATTR_NAMESPACE_USER, 0, 0);
                } else {
                    status = ::extattr_list_file(pth.c_str(), EXTATTR_NAMESPACE_USER, 0, 0);
                }
                if (status < 0) { return detail::stringvec_t{}; }
                
                /// NEEDED on FreeBSD (no ending null):
                attrbuffer = detail::attrbuf(status + 1);
                attrbuffer.get()[status] = 0;
                
                if (options & attribute::flags::nofollow) {
                    status = ::extattr_list_link(pth.c_str(), EXTATTR_NAMESPACE_USER, attrbuffer.get(), status);
                } else {
                    status = ::extattr_list_file(pth.c_str(), EXTATTR_NAMESPACE_USER, attrbuffer.get(), status);
                }
            
            #elif defined(PXALINUX)
                if (options & attribute::flags::nofollow) {
                    status = ::llistxattr(pth.c_str(), 0, 0);
                } else {
                    status = ::listxattr(pth.c_str(), 0, 0);
                }
                if (status < 0) { return detail::stringvec_t{}; }
                
                /// Don't want to deal with possible status = 0:
                attrbuffer = detail::attrbuf(status + 1);
                
                if (options & attribute::flags::nofollow) {
                    status = ::llistxattr(pth.c_str(), attrbuffer.get(), status);
                } else {
                    status = ::listxattr(pth.c_str(), attrbuffer.get(), status);
                }
            
            #elif defined(__APPLE__)
                if (options & attribute::flags::nofollow) {
                    status = ::listxattr(pth.c_str(), 0, 0, XATTR_NOFOLLOW);
                } else {
                    status = ::listxattr(pth.c_str(), 0, 0, 0);
                }
                if (status < 0) { return detail::stringvec_t{}; }
                
                /// Don't want to deal with possible status = 0:
                attrbuffer = detail::attrbuf(status + 1);
                
                if (options & attribute::flags::nofollow) {
                    status = ::listxattr(pth.c_str(), attrbuffer.get(), status, XATTR_NOFOLLOW);
                } else {
                    status = ::listxattr(pth.c_str(), attrbuffer.get(), status, 0);
                }
                
            #endif
            
            if (!attrbuffer.get()) { return detail::stringvec_t{}; }
            char* buffer_start = attrbuffer.get();
            
            detail::stringvec_t out{};
            
            #if defined(__FreeBSD__)
                char* buffer_pos = attrbuffer.get();
                char* cp = attrbuffer.get();
                std::size_t len;
                while (cp < buffer_pos + status + 1) {
                    len = *cp;
                    *cp = 0;
                    cp += len + 1;
                }
                buffer_start = buffer_pos + 1;
                *cp = 0; // don't forget, we allocated one more
            #endif
            
            if (status > 0) {
                int pos = 0;
                while (pos < status) {
                    std::string n = std::string(buffer_start + pos);
                    std::string pxn = detail::pxaname(n);
                    if (pxn != detail::nullstring) {
                        out.push_back(pxn);
                    }
                    pos += n.length() + 1;
                }
            }
            
            return out;
        }
        
        
        accessor_t::accessvec_t accessor_t::list(filesystem::path const& pth,
                                                 attribute::flags options,
                                                 attribute::ns domain) {
            std::string pthstr = pth.str();
            detail::stringvec_t strings = attribute::list(pthstr, options, domain);
            accessor_t::accessvec_t out{};
            if (!strings.empty()) {
                std::for_each(strings.begin(),
                              strings.end(),
                          [&](std::string const& attrname) { out.emplace_back(pthstr, attrname); });
            }
            return out;
        }
        
        accessor_t::accessvec_t accessor_t::list(std::string const& pth,
                                                 attribute::flags options,
                                                 attribute::ns domain) {
            detail::stringvec_t strings = attribute::list(pth, options, domain);
            accessor_t::accessvec_t out{};
            if (!strings.empty()) {
                std::for_each(strings.begin(),
                              strings.end(),
                          [&](std::string const& attrname) { out.emplace_back(pth, attrname); });
            }
            return out;
        }
        
        accessor_t::accessor_t(filesystem::path const& pth,
                               std::string const& name,
                               attribute::ns domain)
            :m_pathstring(pth.str())
            ,m_name(name)
            ,m_domain(domain)
            {}
        
        accessor_t::accessor_t(std::string const& pth,
                               std::string const& name,
                               attribute::ns domain)
            :m_pathstring(pth)
            ,m_name(name)
            ,m_domain(domain)
            {}
        
        accessor_t::accessor_t(accessor_t const& other)
            :m_pathstring(other.m_pathstring)
            ,m_name(other.m_name)
            ,m_options(other.m_options)
            ,m_domain(other.m_domain)
            {}
            
        accessor_t::accessor_t(accessor_t&& other)
            :m_pathstring(std::move(other.m_pathstring))
            ,m_name(std::move(other.m_name))
            ,m_options(other.m_options)
            ,m_domain(other.m_domain)
            {}
        
        accessor_t::~accessor_t() {}
        
        std::string accessor_t::get(attribute::flags o,
                                    attribute::ns d) const {
            return attribute::get(pathstring(), name(),
                                  options(), domain());
        }
        
        bool accessor_t::set(std::string const& value,
                             attribute::flags o,
                             attribute::ns d) const {
            return attribute::set(pathstring(), name(), value,
                                  options(), domain());
        }
        
        bool accessor_t::del(attribute::flags o,
                             attribute::ns d) const {
            return attribute::del(pathstring(), name(),
                                  options(), domain());
        }
        
        std::string accessor_t::pathstring() const {
            return m_pathstring;
        }
        
        std::string accessor_t::pathstring(std::string const& newpathstring) {
            m_pathstring = std::string(newpathstring);
            return m_pathstring;
        }
        
        std::string accessor_t::name() const {
            return m_name;
        }
        
        std::string accessor_t::name(std::string const& newname) {
            m_name = std::string(newname);
            return m_name;
        }
        
        attribute::flags accessor_t::options() const {
            return m_options;
        }
        
        attribute::flags accessor_t::options(attribute::flags newflags) {
            m_options = newflags;
            return m_options;
        }
        
        attribute::ns accessor_t::domain() const {
            return m_domain;
        }
        
        attribute::ns accessor_t::domain(attribute::ns newdomain) {
            m_domain = newdomain;
            return m_domain;
        }
        
        accessor_t::operator std::string() const {
            return get();
        }
        
        
    } /// namespace attribute
    
} /// namespace filesystem

#endif /// defined(__FreeBSD__) || defined(PXALINUX) || defined(__APPLE__)