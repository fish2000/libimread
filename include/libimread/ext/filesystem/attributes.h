/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_
#define LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_

#include <memory>
#include <string>
#include <vector>

namespace filesystem {
    
    class path;
    
    namespace attribute {
        
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
            
            using stringvec_t = std::vector<std::string>;
            using attrbuf_t = std::unique_ptr<char[]>;
            
            std::string sysname(std::string const&, attribute::ns domain = attribute::ns::user);
            std::string pxaname(std::string const&, attribute::ns domain = attribute::ns::user);
            
        }
        
        std::string get(std::string const& pth,
                        std::string const& name,
                        attribute::flags options,
                        attribute::ns domain = attribute::ns::user);
        
        bool set(std::string const& pth,
                 std::string const& name,
                 std::string const& value,
                 attribute::flags options,
                 attribute::ns domain = attribute::ns::user);
        
        bool del(std::string const& pth,
                 std::string const& name,
                 attribute::flags options,
                 attribute::ns domain = attribute::ns::user);
        
        detail::stringvec_t list(std::string const& pth,
                                 attribute::flags options,
                                 attribute::ns domain = attribute::ns::user);
        
        class accessor_t {
            
            public:
                using accessvec_t = std::vector<accessor_t>;
                
                static accessvec_t list(filesystem::path const& pth,
                                        attribute::flags options,
                                        attribute::ns domain = attribute::ns::user);
                
                static accessvec_t list(std::string const& pth,
                                        attribute::flags options,
                                        attribute::ns domain = attribute::ns::user);
                
                accessor_t(filesystem::path const& pth,
                           std::string const& name,
                           attribute::ns domain = attribute::ns::user);
                
                accessor_t(std::string const& pth,
                           std::string const& name,
                           attribute::ns domain = attribute::ns::user);
                
                accessor_t(accessor_t const& other);
                accessor_t(accessor_t&& other);
                
                virtual ~accessor_t();
                
                std::string get(attribute::flags options = attribute::flags::none,
                                attribute::ns domain = attribute::ns::user) const;
                
                bool        set(std::string const& value,
                                attribute::flags options = attribute::flags::none,
                                attribute::ns domain = attribute::ns::user) const;
                
                bool        del(attribute::flags options = attribute::flags::none,
                                attribute::ns domain = attribute::ns::user) const;
                
                std::string pathstring() const;
                std::string pathstring(std::string const&);
                std::string name() const;
                std::string name(std::string const&);
                attribute::flags options() const;
                attribute::flags options(attribute::flags);
                attribute::ns domain() const;
                attribute::ns domain(attribute::ns);
                
                operator std::string() const;
            
            private:
                mutable std::string m_pathstring;
                mutable std::string m_name;
                attribute::flags m_options = attribute::flags::none;
                attribute::ns m_domain = attribute::ns::user;
            
        };
        
        bool operator&(ns lhs, ns rhs) {
            return (std::size_t)lhs & (std::size_t)rhs;
        }
        
        bool operator&(flags lhs, flags rhs) {
            return (std::size_t)lhs & (std::size_t)rhs;
        }
        
        
    } /// namespace attribute
    
} /// namespace filesystem

#endif /// LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_
