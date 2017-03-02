/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_STORE_HH_
#define LIBIMREAD_INCLUDE_STORE_HH_

#include <unordered_map>
#include <type_traits>
#include <utility>
#include <string>
#include <vector>

#include <libimread/libimread.hpp>

namespace store {
    
    template <typename Key, typename Mapped,
              typename Value = std::pair<std::add_const_t<Key>, Mapped>>
    class base {
        
        public:
            using key_type = Key;
            using mapped_type = Mapped;
            using value_type = Value;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            
            using reference = std::add_lvalue_reference_t<value_type>;
            using rvalue_reference = std::add_rvalue_reference_t<value_type>;
            using const_reference = std::add_lvalue_reference_t<
                                    std::add_const_t<value_type>>;
            
            using key_reference = std::add_lvalue_reference_t<key_type>;
            using key_const_reference = std::add_lvalue_reference_t<
                                        std::add_const_t<key_type>>;
            using key_rvalue_reference = std::add_rvalue_reference_t<key_type>;
            
            using mapped_reference = std::add_lvalue_reference_t<mapped_type>;
            using mapped_const_reference = std::add_lvalue_reference_t<
                                           std::add_const_t<mapped_type>>;
            using mapped_rvalue_reference = std::add_rvalue_reference_t<mapped_type>;
            
            using container_type = std::unordered_map<key_type, mapped_type>;
        
        public:
            virtual ~base() {}
            virtual key_reference null_v() const {
                static std::string nv("\uFFFF");
                return nv;
            }
            
            virtual bool empty() const = 0;
            virtual size_type size() const = 0;
            virtual size_type max_size() const noexcept = 0;
            
            virtual void clear() = 0;
            virtual bool insert(rvalue_reference) = 0;
            virtual bool emplace(reference) { return false; }
            virtual size_type erase(key_const_reference) = 0;
            
            virtual mapped_reference at(key_const_reference) = 0;
            virtual mapped_const_reference at(key_const_reference) const = 0;
            virtual mapped_reference operator[](key_const_reference) = 0;
            virtual mapped_reference operator[](key_rvalue_reference) { return null_v(); }
            virtual size_type count(key_const_reference) const = 0;
    };
    
    class stringmap : public base<std::string, std::string> {
        
        public:
            using base_t = base<std::string, std::string>;
            using stringvec_t = std::vector<std::string>;
            using stringmap_t = std::unordered_map<std::string, std::string>;
        
        public:
            virtual std::string&       get(std::string const& key) = 0;
            virtual std::string const& get(std::string const& key) const = 0;
            virtual bool set(std::string const& key, std::string const& value) = 0;
            virtual bool del(std::string const& key) = 0;
            virtual std::size_t count() const = 0;
            virtual stringvec_t list() const = 0;
            
            virtual stringmap_t mapping() const;
        
        public:
            virtual bool empty() const;
            virtual std::size_t size() const;
            virtual std::size_t max_size() const noexcept;
            
            virtual void clear();
            virtual bool insert(std::pair<const std::string, std::string>&& item);
            virtual std::size_t erase(std::string const& key);
            virtual std::string& at(std::string const& key);
            virtual std::string const& at(std::string const& key) const;
            virtual std::string& operator[](std::string const& key);
            virtual std::string& operator[](std::string&& key);
            virtual std::size_t count(std::string const& key) const;
        
        protected:
            stringmap_t cache;
    
    };
    
} /// namespace store

#endif /// LIBIMREAD_INCLUDE_STORE_HH_