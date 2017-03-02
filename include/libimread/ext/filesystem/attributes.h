/// Copyright 2017 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Inspired By / Modeled After: `pxattr` by @edenzik
///     ... q.v. https://github.com/edenzik/pxattr sub.

#ifndef LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_
#define LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_

#include <memory>
#include <string>

#include <libimread/libimread.hpp>

#define ENUMBASE(enumclass) (enumclass)0

namespace filesystem {
    
    class path;
    
    namespace attribute {
        
        enum struct ns : std::size_t;
        enum struct flags : std::size_t;
        
        namespace detail {
            
            using stringvec_t = std::vector<std::string>;
            using attrbuf_t = std::unique_ptr<char[]>;
            
            #if defined(PXALINUX) || defined(COMPAT1)
            static const std::string userstring("user.");
            #else
            static const std::string userstring("");
            #endif
            static const std::string nullstring{ NULL_STR };
            
            std::string sysname(std::string const&, attribute::ns domain = ENUMBASE(attribute::ns));
            std::string pxaname(std::string const&, attribute::ns domain = ENUMBASE(attribute::ns));
            
        }
        
        std::string get(std::string const& pth,
                        std::string const& name,
                        attribute::flags options = ENUMBASE(attribute::flags),
                        attribute::ns domain = ENUMBASE(attribute::ns));
        
        bool set(std::string const& pth,
                 std::string const& name,
                 std::string const& value,
                 attribute::flags options = ENUMBASE(attribute::flags),
                 attribute::ns domain = ENUMBASE(attribute::ns));
        
        bool del(std::string const& pth,
                 std::string const& name,
                 attribute::flags options = ENUMBASE(attribute::flags),
                 attribute::ns domain = ENUMBASE(attribute::ns));
        
        detail::stringvec_t list(std::string const& pth,
                                 attribute::flags options = ENUMBASE(attribute::flags),
                                 attribute::ns domain = ENUMBASE(attribute::ns));
        
        int count(std::string const& pth,
                  attribute::flags options = ENUMBASE(attribute::flags),
                  attribute::ns domain = ENUMBASE(attribute::ns));
        
        std::string fdget(int descriptor,
                          std::string const& name,
                          attribute::flags options = ENUMBASE(attribute::flags),
                          attribute::ns domain = ENUMBASE(attribute::ns));
        
        bool fdset(int descriptor,
                   std::string const& name,
                   std::string const& value,
                   attribute::flags options = ENUMBASE(attribute::flags),
                   attribute::ns domain = ENUMBASE(attribute::ns));
        
        bool fddel(int descriptor,
                   std::string const& name,
                   attribute::flags options = ENUMBASE(attribute::flags),
                   attribute::ns domain = ENUMBASE(attribute::ns));
        
        detail::stringvec_t fdlist(int descriptor,
                                   attribute::flags options = ENUMBASE(attribute::flags),
                                   attribute::ns domain = ENUMBASE(attribute::ns));
        
        int fdcount(int descriptor,
                    attribute::flags options = ENUMBASE(attribute::flags),
                    attribute::ns domain = ENUMBASE(attribute::ns));
        
        class accessor_t {
            
            public:
                using accessvec_t = std::vector<accessor_t>;
                
                static accessvec_t list(int descriptor,
                                        attribute::flags options,
                                        attribute::ns domain = ENUMBASE(attribute::ns));
                
                static accessvec_t list(filesystem::path const& pth,
                                        attribute::flags options,
                                        attribute::ns domain = ENUMBASE(attribute::ns));
                
                static accessvec_t list(std::string const& pth,
                                        attribute::flags options,
                                        attribute::ns domain = ENUMBASE(attribute::ns));
                
                accessor_t(int descriptor,
                           std::string const& name,
                           attribute::ns domain = ENUMBASE(attribute::ns));
                
                accessor_t(filesystem::path const& pth,
                           std::string const& name,
                           attribute::ns domain = ENUMBASE(attribute::ns));
                
                accessor_t(std::string const& pth,
                           std::string const& name,
                           attribute::ns domain = ENUMBASE(attribute::ns));
                
                accessor_t(accessor_t const& other);
                accessor_t(accessor_t&& other);
                
                virtual ~accessor_t();
                
                std::string get(attribute::flags options = ENUMBASE(attribute::flags),
                                attribute::ns domain = ENUMBASE(attribute::ns)) const;
                
                bool        set(std::string const& value,
                                attribute::flags options = ENUMBASE(attribute::flags),
                                attribute::ns domain = ENUMBASE(attribute::ns)) const;
                
                bool        del(attribute::flags options = ENUMBASE(attribute::flags),
                                attribute::ns domain = ENUMBASE(attribute::ns)) const;
                
                std::string pathstring() const;
                std::string pathstring(std::string const&);
                std::string name() const;
                std::string name(std::string const&);
                int descriptor() const;
                int descriptor(int);
                attribute::flags options() const;
                attribute::flags options(attribute::flags);
                attribute::ns domain() const;
                attribute::ns domain(attribute::ns);
                
                operator std::string() const;
            
            private:
                mutable std::string m_pathstring;
                mutable std::string m_name;
                mutable int m_descriptor = -1;
                attribute::flags m_options = ENUMBASE(attribute::flags);
                attribute::ns m_domain = ENUMBASE(attribute::ns);
            
        };
        
    } /// namespace attribute
    
} /// namespace filesystem

#undef ENUMBASE

#endif /// LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_
