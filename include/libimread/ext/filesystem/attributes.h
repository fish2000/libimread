/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_
#define LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_

#include <memory>
#include <string>
#include <vector>

#define ENUM_DEFAULT(enumclass) (enumclass)0

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
            static const std::string nullstring("\uFFFF");
            
            std::string sysname(std::string const&, attribute::ns domain = ENUM_DEFAULT(attribute::ns));
            std::string pxaname(std::string const&, attribute::ns domain = ENUM_DEFAULT(attribute::ns));
            
        }
        
        std::string get(std::string const& pth,
                        std::string const& name,
                        attribute::flags options = ENUM_DEFAULT(attribute::flags),
                        attribute::ns domain = ENUM_DEFAULT(attribute::ns));
        
        bool set(std::string const& pth,
                 std::string const& name,
                 std::string const& value,
                 attribute::flags options = ENUM_DEFAULT(attribute::flags),
                 attribute::ns domain = ENUM_DEFAULT(attribute::ns));
        
        bool del(std::string const& pth,
                 std::string const& name,
                 attribute::flags options = ENUM_DEFAULT(attribute::flags),
                 attribute::ns domain = ENUM_DEFAULT(attribute::ns));
        
        detail::stringvec_t list(std::string const& pth,
                                 attribute::flags options = ENUM_DEFAULT(attribute::flags),
                                 attribute::ns domain = ENUM_DEFAULT(attribute::ns));
        
        class accessor_t {
            
            public:
                using accessvec_t = std::vector<accessor_t>;
                
                static accessvec_t list(filesystem::path const& pth,
                                        attribute::flags options,
                                        attribute::ns domain = ENUM_DEFAULT(attribute::ns));
                
                static accessvec_t list(std::string const& pth,
                                        attribute::flags options,
                                        attribute::ns domain = ENUM_DEFAULT(attribute::ns));
                
                accessor_t(filesystem::path const& pth,
                           std::string const& name,
                           attribute::ns domain = ENUM_DEFAULT(attribute::ns));
                
                accessor_t(std::string const& pth,
                           std::string const& name,
                           attribute::ns domain = ENUM_DEFAULT(attribute::ns));
                
                accessor_t(accessor_t const& other);
                accessor_t(accessor_t&& other);
                
                virtual ~accessor_t();
                
                std::string get(attribute::flags options = ENUM_DEFAULT(attribute::flags),
                                attribute::ns domain = ENUM_DEFAULT(attribute::ns)) const;
                
                bool        set(std::string const& value,
                                attribute::flags options = ENUM_DEFAULT(attribute::flags),
                                attribute::ns domain = ENUM_DEFAULT(attribute::ns)) const;
                
                bool        del(attribute::flags options = ENUM_DEFAULT(attribute::flags),
                                attribute::ns domain = ENUM_DEFAULT(attribute::ns)) const;
                
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
                attribute::flags m_options = ENUM_DEFAULT(attribute::flags);
                attribute::ns m_domain = ENUM_DEFAULT(attribute::ns);
            
        };
        
    } /// namespace attribute
    
} /// namespace filesystem

#undef ENUM_DEFAULT

#endif /// LIBIMREAD_EXT_FILESYSTEM_ATTRIBUTES_H_
