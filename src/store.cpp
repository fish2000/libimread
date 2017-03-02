/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/store.hh>

namespace store {
    
    stringmapper::stringmap_t stringmapper::mapping() const {
        return stringmapper::stringmap_t(cache);
    }
    
    stringmapper::~stringmapper() {}
    
    bool stringmapper::empty() const {
        return count() == 0;
    }
    
    std::size_t stringmapper::size() const {
        return count();
    }
    
    std::size_t stringmapper::max_size() const noexcept {
        return stringvec_t().max_size();
    }
    
    void stringmapper::clear() {
        for (std::string const& key : list()) {
            del(key);
        }
    }
    
    bool stringmapper::insert(std::pair<const std::string, std::string>&& item) {
        del(item.second);
        return set(item.first, item.second);
    }
    
    std::size_t stringmapper::erase(std::string const& key) {
        return del(key);
    }
    
    std::string& stringmapper::at(std::string const& key) {
        return get(key);
    }
    
    std::string const& stringmapper::at(std::string const& key) const {
        return get(key);
    }
    
    std::string& stringmapper::operator[](std::string const& key) {
        return get(key);
    }
    
    std::string& stringmapper::operator[](std::string&& key) {
        return get(key);
    }
    
    std::size_t stringmapper::count(std::string const& key) const {
        return static_cast<std::size_t>(get(key) == stringmapper::base_t::null_value());
    }
    
    bool xattrmap::can_store() const noexcept { return true; }
    
    std::string& xattrmap::get(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            std::string val(xattr(key));
            if (val != stringmapper::base_t::null_value()) {
                cache[key] = val;
                return cache[key];
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    std::string const& xattrmap::get(std::string const& key) const {
        if (cache.find(key) != cache.end()) {
            return cache[key];
        } else {
            std::string val(xattr(key));
            if (val != stringmapper::base_t::null_value()) {
                cache[key] = val;
                return cache[key];
            }
            return stringmapper::base_t::null_value();
        }
    }
    
    bool xattrmap::set(std::string const& key, std::string const& value) {
        cache[key] = value;
        return xattr(key, value) == key;
    }
    
    bool xattrmap::del(std::string const& key) {
        if (cache.find(key) != cache.end()) {
            cache.erase(key);
        }
        return xattr(key, stringmapper::base_t::null_value()) == stringmapper::base_t::null_value();
    }
    
    std::size_t xattrmap::count() const {
        return xattrcount();
    }
    
    stringmapper::stringvec_t xattrmap::list() const {
        return xattrs();
    }
    
} /// namespace store
