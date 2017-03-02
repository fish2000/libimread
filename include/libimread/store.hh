/// Copyright 2012-2017 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INCLUDE_STORE_HH_
#define LIBIMREAD_INCLUDE_STORE_HH_

#include <unordered_map>
#include <type_traits>
#include <utility>
#include <string>
#include <vector>

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
            
            virtual key_reference null_key() const {
                static key_type nk{ 0 };
                return nk;
            }
            
            virtual mapped_reference null_value() const {
                static mapped_type nv{ 0 };
                return nv;
            }
            
            virtual bool can_store() const noexcept { return false; }
            
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
            virtual mapped_reference operator[](key_rvalue_reference) { return null_value(); }
            virtual size_type count(key_const_reference) const = 0;
    };
    
    class stringmapper : public base<std::string, std::string> {
        
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
            virtual std::string mapping_json() const;
        
        public:
            virtual ~stringmapper();
            virtual key_reference null_key() const override;
            virtual mapped_reference null_value() const override;
            
            virtual bool empty() const override;
            virtual std::size_t size() const override;
            virtual std::size_t max_size() const noexcept override;
            
            virtual void clear() override;
            virtual bool insert(std::pair<const std::string, std::string>&& item) override;
            virtual std::size_t erase(std::string const& key) override;
            virtual std::string& at(std::string const& key) override;
            virtual std::string const& at(std::string const& key) const override;
            virtual std::string& operator[](std::string const& key) override;
            virtual std::string& operator[](std::string&& key) override;
            virtual std::size_t count(std::string const& key) const override;
        
        protected:
            mutable stringmap_t cache;
    
    };
    
    template <typename ...Types>
    struct is_stringmapper : std::conditional_t<
                             std::is_base_of<store::stringmapper,
                                             typename std::common_type<Types...>::type>::value,
                             std::true_type,
                             std::false_type>
    {};
    
    template <typename ...Types>
    constexpr bool is_stringmapper_v = is_stringmapper<Types...>::value;
    
    template <typename T, typename U>
    void value_copy(T&& from, U&& to) {
        static_assert(store::is_stringmapper_v<T, U>,
                      "store::value_copy() operands must derive from store::stringmapper");
        stringmapper::stringvec_t froms(std::forward<T>(from).list());
        if (!froms.empty()) {
            for (std::string const& name : froms) { std::forward<U>(to).set(name,
                                                    std::forward<T>(from).get(name)); }
        }
    }
    
    template <typename T, typename U>
    void prefix_copy(T&& from, U&& to, std::string const& prefix = "prefix",
                                       std::string const& sep = ":") {
        static_assert(store::is_stringmapper_v<T, U>,
                      "store::prefix_copy() operands must derive from store::stringmapper");
        stringmapper::stringvec_t froms(std::forward<T>(from).list());
        if (!froms.empty()) {
            for (std::string const& name : froms) { std::forward<U>(to).set(prefix + sep + name,
                                                    std::forward<T>(from).get(name)); }
        }
    }
    
    class xattrmap : public stringmapper {
        
        public:
            virtual bool can_store() const noexcept override;
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            X update(T&& from) {
                store::value_copy(std::forward<T>(from), *this);
            }
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            X update(T&& from, std::string const& prefix,
                               std::string const& sep = ":") {
                store::prefix_copy(std::forward<T>(from), *this, prefix, sep);
            }
            
        
        public:
            /// xattr API
            virtual std::string xattr(std::string const&) const = 0;
            virtual std::string xattr(std::string const&, std::string const&) const = 0;
            virtual int xattrcount() const = 0;
            virtual stringvec_t xattrs() const = 0;
        
        public:
            /// implementation of the stringmapper API, in terms of xattr()/xattrcount()/xattrs()
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
        
    };
    
    class stringmap : public stringmapper {
        
        public:
            virtual bool can_store() const noexcept override;
        
        public:
            stringmap() noexcept;                   /// default constructor
            explicit stringmap(std::string const&); /// decode from JSON string
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            explicit stringmap(T&& from) noexcept
                :stringmap()
                {
                    store::value_copy(std::forward<T>(from), *this);
                }
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            explicit stringmap(T&& from, std::string const& prefix,
                                         std::string const& sep = ":") noexcept
                :stringmap()
                {
                    store::prefix_copy(std::forward<T>(from), *this, prefix, sep);
                }
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            X update(T&& from) {
                store::value_copy(std::forward<T>(from), *this);
            }
            
            template <typename T,
                      typename X = typename std::enable_if_t<
                                            store::is_stringmapper_v<T>, void>>
            X update(T&& from, std::string const& prefix,
                               std::string const& sep = ":") {
                store::prefix_copy(std::forward<T>(from), *this, prefix, sep);
            }
        
        public:
            /// implementation of the stringmapper API, in terms of std::unordered_map<…> API
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
        
    };
    
    
} /// namespace store

#endif /// LIBIMREAD_INCLUDE_STORE_HH_